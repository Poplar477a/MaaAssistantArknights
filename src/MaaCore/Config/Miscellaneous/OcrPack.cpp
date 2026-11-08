#ifndef __ANDROID__

#include "OcrPack.h"

#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <thread>
#include <unordered_map>

#include "Config/WebGpuDevice.h"
#include "MaaUtils/NoWarningCV.hpp"
MAA_SUPPRESS_CV_WARNINGS_BEGIN
#include "fastdeploy/vision/ocr/ppocr/dbdetector.h"
#include "fastdeploy/vision/ocr/ppocr/ppocr_v6.h"
#include "fastdeploy/vision/ocr/ppocr/recognizer.h"
MAA_SUPPRESS_CV_WARNINGS_END

#include "Utils/Demangle.hpp"
#include "Utils/Logger.hpp"
#include "Utils/Platform.hpp"
#include "Utils/StringMisc.hpp"
#include <map>
#include <mutex>
#include <ranges>
#include <string>

#ifdef MAA_WITH_OPENVINO
#include <cmath>
#include <cstring>
#include <fstream>
#if defined(_MSC_VER)
#pragma warning(push)
// ov::Tensor::data<T>() keeps the mutable-pointer overload deprecated until 2026.0
// while MaaCore builds with warnings as errors.
#pragma warning(disable: 4996)
#endif
#include <openvino/openvino.hpp>

#include "db_postprocess.h"
#endif

namespace asst
{

#ifdef MAA_WITH_OPENVINO
namespace ocr_ov
{
// One OpenVINO Core per process: separate Core instances load the same device
// plugin repeatedly and were observed to crash the NPU plugin under load.
ov::Core& shared_core()
{
    // Intentionally leaked: OcrPack singletons may outlive a function-local
    // static during process teardown, and the ORT side leaks its sessions too.
    static ov::Core* core = new ov::Core();
    return *core;
}

// The NPU plugin is not safe for concurrent inference requests; WordOcr and
// CharOcr can be driven from different pipeline stages, so serialize them.
std::mutex& npu_mutex()
{
    static std::mutex mutex;
    return mutex;
}

constexpr int DetHeight = 544;
constexpr int DetWidth = 960;
constexpr int RecHeight = 48;
// Two recognition buckets: most UI lines fit 320; longer lines are routed to
// 960 instead of being clipped. Both are compiled lazily on first use.
constexpr int RecWidthShort = 320;
constexpr int RecWidthLong = 960;

constexpr float DetBinThreshold = 0.2F;
constexpr float BoxThreshold = 0.45F;
constexpr float UnclipRatio = 1.4F;
constexpr int MaxCandidates = 3000;

std::vector<std::string> load_keys(const std::filesystem::path& path)
{
    std::vector<std::string> keys;
    std::ifstream ifs(platform::path_to_utf8_string(path), std::ios::binary);
    if (!ifs) {
        return keys;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        keys.emplace_back(std::move(line));
    }
    return keys;
}

std::string
    ctc_decode(const float* logits, int steps, int classes, const std::vector<std::string>& keys, float& out_score)
{
    std::string text;
    float score_sum = 0.0F;
    int score_count = 0;
    int prev = -1;

    for (int step = 0; step < steps; ++step) {
        const float* row = logits + static_cast<std::size_t>(step) * static_cast<std::size_t>(classes);
        int best = 0;
        float best_value = row[0];
        for (int c = 1; c < classes; ++c) {
            if (row[c] > best_value) {
                best_value = row[c];
                best = c;
            }
        }
        if (best != 0 && best != prev) {
            const int index = best - 1;
            if (index >= 0 && index < static_cast<int>(keys.size())) {
                text += keys[index];
                score_sum += best_value;
                ++score_count;
            }
        }
        prev = best;
    }

    out_score = score_count > 0 ? score_sum / static_cast<float>(score_count) : 0.0F;
    return text;
}

void fill_det_input(const cv::Mat& image, std::vector<float>& dst, float& out_scale)
{
    const float scale = std::min(
        static_cast<float>(DetWidth) / static_cast<float>(image.cols),
        static_cast<float>(DetHeight) / static_cast<float>(image.rows));
    const int resized_w = std::max(1, static_cast<int>(static_cast<float>(image.cols) * scale));
    const int resized_h = std::max(1, static_cast<int>(static_cast<float>(image.rows) * scale));

    cv::Mat canvas(DetHeight, DetWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(0, 0, resized_w, resized_h)));

    cv::Mat float_image;
    canvas.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    cv::Mat normalised = (float_image - cv::Scalar(0.485, 0.456, 0.406)) / cv::Scalar(0.229, 0.224, 0.225);

    const std::size_t plane = static_cast<std::size_t>(DetHeight) * static_cast<std::size_t>(DetWidth);
    dst.resize(plane * 3);
    std::vector<cv::Mat> channels(3);
    cv::split(normalised, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(dst.data() + static_cast<std::size_t>(c) * plane, channels[c].ptr<float>(), sizeof(float) * plane);
    }
    out_scale = scale;
}

int rec_scaled_width(const cv::Mat& crop)
{
    return std::max(1, static_cast<int>(static_cast<float>(crop.cols) * RecHeight / static_cast<float>(crop.rows)));
}

void fill_rec_input(const cv::Mat& crop, std::vector<float>& dst, int bucket_width)
{
    const float scale = static_cast<float>(RecHeight) / static_cast<float>(crop.rows);
    cv::Mat resized_crop;
    cv::resize(
        crop,
        resized_crop,
        cv::Size(std::max(1, static_cast<int>(static_cast<float>(crop.cols) * scale)), RecHeight),
        0,
        0,
        cv::INTER_LINEAR);

    cv::Mat padded(RecHeight, bucket_width, CV_8UC3, cv::Scalar(0, 0, 0));
    const int copy_w = std::min(bucket_width, resized_crop.cols);
    resized_crop(cv::Rect(0, 0, copy_w, RecHeight)).copyTo(padded(cv::Rect(0, 0, copy_w, RecHeight)));

    cv::Mat float_image;
    padded.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    float_image = (float_image - 0.5F) / 0.5F;

    const std::size_t plane = static_cast<std::size_t>(RecHeight) * static_cast<std::size_t>(bucket_width);
    dst.resize(plane * 3);
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(dst.data() + static_cast<std::size_t>(c) * plane, channels[c].ptr<float>(), sizeof(float) * plane);
    }
}

std::shared_ptr<ov::Model> read_static_model(ov::Core& core, const std::filesystem::path& path, const ov::Shape& target)
{
    std::shared_ptr<ov::Model> model = core.read_model(platform::path_to_utf8_string(path));
    const auto input = model->input();
    if (input.get_partial_shape().is_static() && input.get_shape() == target) {
        return model;
    }
    try {
        model->reshape({ { input.get_any_name(), target } });
    }
    catch (const std::exception& e) {
        LogError << __FUNCTION__ << "reshape rejected by" << path.filename().string() << e.what();
    }
    return model;
}

// EXECUTION_DEVICES is the authoritative answer to "which device is this
// compiled model actually running on"; FULL_DEVICE_NAME identifies the
// physical unit. NPU returns a scalar string for EXECUTION_DEVICES while CPU
// returns a string vector, so read it as ov::Any instead of the typed property.
std::string any_to_string(const ov::Any& value)
{
    if (value.is<std::string>()) {
        return value.as<std::string>();
    }
    if (value.is<std::vector<std::string>>()) {
        std::string joined;
        for (const auto& item : value.as<std::vector<std::string>>()) {
            if (!joined.empty()) {
                joined += ",";
            }
            joined += item;
        }
        return joined;
    }
    try {
        return value.as<std::string>();
    }
    catch (...) {
        return "?";
    }
}

std::string device_summary(const ov::CompiledModel& model, const std::string& device)
{
    std::string execution;
    try {
        execution = any_to_string(model.get_property("EXECUTION_DEVICES"));
    }
    catch (const std::exception& e) {
        execution = std::string("unknown: ") + e.what();
    }

    std::string full_name;
    try {
        full_name = shared_core().get_property(device, "FULL_DEVICE_NAME").as<std::string>();
    }
    catch (...) {
        full_name = "?";
    }

    return execution + " [" + full_name + "]";
}
} // namespace ocr_ov
#endif

struct OcrPack::Impl
{
    std::unique_ptr<fastdeploy::vision::ocr::DBDetector> det;
    std::unique_ptr<fastdeploy::vision::ocr::Recognizer> rec;
    std::unique_ptr<fastdeploy::pipeline::PPOCRv6> ocr;

    std::filesystem::path det_model_path;
    std::filesystem::path rec_model_path;
    std::filesystem::path rec_label_path;

    // WebGPU 的 Dawn instance/device 句柄，生命周期需覆盖会话创建（见 WebGpuDevice.h）
    std::unordered_map<std::string, std::string> webgpu_options;

#ifdef MAA_WITH_OPENVINO
    ov::CompiledModel ov_det_model;
    ov::InferRequest ov_det_req;
    bool ov_det_ready = false;

    struct RecBucket
    {
        ov::CompiledModel model;
        ov::InferRequest request;
    };

    std::map<int, RecBucket> ov_rec_buckets;
    bool ov_long_bucket_failed = false;

    std::string ov_device;
    ov::AnyMap ov_props;
    std::vector<std::string> ov_keys;
    std::vector<float> ov_det_in;
    std::vector<float> ov_rec_in;
#endif
};

OcrPack::OcrPack() :
    m_impl(std::make_unique<Impl>())
{
}

OcrPack::~OcrPack()
{
    LogTraceFunction;
    if (m_gpu_active) {
        // FIXME: leak fastdeploy objects to avoid crash (double free?)
        (void)m_impl->det.release();
        (void)m_impl->rec.release();
        (void)m_impl->ocr.release();
    }

#ifdef MAA_WITH_OPENVINO
    if (m_ov_active) {
        // Releasing the NPU models while the process tears down crashes inside
        // openvino_intel_npu_plugin.dll (AV in its destructor path). The OS
        // reclaims the memory anyway; keep the objects alive to exit cleanly.
        (void)new Impl(std::move(*m_impl));
    }
#endif
}

bool OcrPack::load(const std::filesystem::path& path)
{
    LogTraceFunction;
    Log.info("load", path.lexically_relative(UserDir.get()));

#ifdef MAA_WITH_OPENVINO
    // Model or language-table paths may have changed; drop the compiled models
    // and the character table so the next check_and_load rebuilds them.
    release_openvino();
    m_impl->ov_keys.clear();
#endif

    using namespace asst::utils::path_literals;
    const auto det_dir = path / "det"_p;
    const auto det_model_file = det_dir / "inference.onnx"_p;

    if (std::filesystem::exists(det_model_file) && m_impl->det_model_path != det_model_file) {
        m_impl->det_model_path = det_model_file;
        m_impl->det = nullptr;
    }
    else if (m_impl->det_model_path.empty()) {
        const auto fallback_det_file = path.parent_path() / "PaddleOCR" / "det" / "inference.onnx"_p;
        if (std::filesystem::exists(fallback_det_file)) {
            m_impl->det_model_path = fallback_det_file;
            m_impl->det = nullptr;
        }
    }

    const auto rec_dir = path / "rec"_p;
    const auto rec_model_file = rec_dir / "inference.onnx"_p;
    const auto rec_label_file = rec_dir / "keys.txt"_p;

    if (std::filesystem::exists(rec_model_file) && m_impl->rec_model_path != rec_model_file) {
        m_impl->rec_model_path = rec_model_file;
        m_impl->rec = nullptr;
    }
    if (std::filesystem::exists(rec_label_file) && m_impl->rec_label_path != rec_label_file) {
        m_impl->rec_label_path = rec_label_file;
        m_impl->rec = nullptr;
    }

    if (m_impl->det && m_impl->rec) {
        m_impl->ocr = std::make_unique<fastdeploy::pipeline::PPOCRv6>(m_impl->det.get(), m_impl->rec.get());
    }

    const bool paths_ready =
        !m_impl->det_model_path.empty() && !m_impl->rec_model_path.empty() && !m_impl->rec_label_path.empty();
    if (!paths_ready) {
        return false;
    }

    // WebGPU 的 shader 编译与 OpenVINO NPU 的设备编译都发生在第一次推理上。这里在资源
    // 加载阶段就把模型建好并预热（check_and_load 内部会预热），避免这笔开销落在第一次识别上。
    // 初始化失败时 check_and_load 会退回 CPU，不把整次资源加载标成失败。
    // 其它后端保持原来的懒加载行为。
    if (m_gpu_selector && (m_gpu_selector->backend() == InferenceBackend::WebGPU ||
                           m_gpu_selector->backend() == InferenceBackend::OpenVINO)) {
        return check_and_load();
    }

    return true;
}

OcrPack::ResultsVec OcrPack::recognize(const cv::Mat& image, bool without_det, const std::optional<Rect>& base_roi)
{
    if (!check_and_load()) {
        Log.error(__FUNCTION__, "check_and_load failed");
        return {};
    }

#ifdef MAA_WITH_OPENVINO
    if (m_ov_active) {
        return recognize_openvino(image, without_det, base_roi);
    }
#endif

    fastdeploy::vision::OCRResult ocr_result;

    // 注意：使用 DirectML 时，调试器下会出现大量 _com_error 异常
    // 不影响程序运行但会导致调试器频繁中断，造成响应停顿
    // 即使禁用 C++ 的 "_com_error" 异常中断，调试器仍可能频繁中断
    // 推荐在挂调试器的情况下禁用 GPU 推理
    // 或等待两轮 Predict 卡个几十秒之后正常运行
    auto start_time = std::chrono::steady_clock::now();
    if (!without_det) {
        m_impl->ocr->Predict(image, &ocr_result);
    }
    else {
        std::string rec_text;
        float rec_score = 0;
        m_impl->rec->Predict(image, &rec_text, &rec_score);
        ocr_result.text.emplace_back(std::move(rec_text));
        ocr_result.rec_scores.emplace_back(rec_score);
    }

#ifdef ASST_DEBUG
    cv::Mat draw = image.clone();
#endif

    ResultsVec raw_results;
    for (size_t i = 0; i != ocr_result.text.size(); ++i) {
        // the box rect like ↓
        // 0 - 1
        // 3 - 2
        Rect det_rect;
        if (!without_det && i < ocr_result.boxes.size()) {
            const auto& box = ocr_result.boxes.at(i);
            int x_collect[] = { box[0], box[2], box[4], box[6] };
            int y_collect[] = { box[1], box[3], box[5], box[7] };
            auto [left, right] = std::ranges::minmax(x_collect);
            auto [top, bottom] = std::ranges::minmax(y_collect);
            det_rect = Rect(left, top, right - left, bottom - top);
        }
        else {
            det_rect = Rect(0, 0, image.cols, image.rows);
        }

#ifdef ASST_DEBUG
        cv::rectangle(draw, make_rect<cv::Rect>(det_rect), cv::Scalar(0, 0, 255), 2);
#endif
        raw_results.emplace_back(Result(det_rect, ocr_result.rec_scores.at(i), std::move(ocr_result.text.at(i))));
    }

    auto costs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
    std::string class_type = utils::demangle(typeid(*this).name());
    if (!base_roi) {
        Log.trace(class_type, raw_results, without_det ? "by OCR Rec" : "by OCR Pipeline", ", cost", costs, "ms");
    }
    else {
        std::string output = "[";
        for (size_t i = 0; i < raw_results.size(); ++i) {
            if (i != 0) {
                output += ", ";
            }
            output += std::format(
                "{{ text: {}, rect: [ {} ({}), {} ({}), {}, {} ], score: {:.6f} }}",
                raw_results[i].text,
                raw_results[i].rect.x,
                raw_results[i].rect.x + base_roi->x,
                raw_results[i].rect.y,
                raw_results[i].rect.y + base_roi->y,
                raw_results[i].rect.width,
                raw_results[i].rect.height,
                raw_results[i].score);
        }
        output += "]";
        Log.trace(class_type, output, without_det ? "by OCR Rec" : "by OCR Pipeline", ", cost", costs, "ms");
    }
    return raw_results;
}

bool OcrPack::pipeline_ready() const
{
#ifdef MAA_WITH_OPENVINO
    if (m_ov_active) {
        return m_impl->ov_det_ready && !m_impl->ov_rec_buckets.empty();
    }
#endif
    return m_impl->det && m_impl->det->Initialized() && m_impl->rec && m_impl->rec->Initialized() && m_impl->ocr &&
           m_impl->ocr->Initialized();
}

void OcrPack::release_models()
{
#ifdef MAA_WITH_OPENVINO
    release_openvino();
#endif
    // GPU 会话在进程退出时销毁会和 ORT 双重释放，和析构函数一样先泄漏。
    // 运行期退回 CPU 时也走这里，避免 reset() 立刻把刚建失败的 GPU 对象析构掉。
    if (m_gpu_active) {
        (void)m_impl->det.release();
        (void)m_impl->rec.release();
        (void)m_impl->ocr.release();
        return;
    }

    m_impl->det.reset();
    m_impl->rec.reset();
    m_impl->ocr.reset();
}

bool OcrPack::check_and_load()
{
    if (pipeline_ready()) {
        return true;
    }

    LogTraceFunction;

    if (create_pipeline(false)) {
        return true;
    }

    // 只在 GPU 会话已经配上、但建模型失败时再试一次 CPU。
    // 设备解析失败时 create_pipeline 自己就会落到 CPU，这里不再重试。
    if (!m_gpu_active) {
        return false;
    }

    Log.error(__FUNCTION__, "| GPU OCR init failed, falling back to CPU");
    m_gpu_active = false;
    return create_pipeline(true);
}

bool OcrPack::create_pipeline(bool force_cpu)
{
    // 先按当前 m_gpu_active 释放上一轮残留。GPU 对象在这里被泄漏，
    // 随后标志清掉，本轮从 CPU 状态开始配。
    release_models();
    m_gpu_active = false;

    int logical = std::max(1u, std::thread::hardware_concurrency());
    int cpu_threads;
    if (logical <= 2) {
        cpu_threads = 1;
    }
    else if (logical <= 4) {
        cpu_threads = 2;
    }
    else if (logical <= 12) {
        cpu_threads = 3;
    }
    else {
        cpu_threads = 4;
    }

    fastdeploy::RuntimeOption det_option;
    fastdeploy::RuntimeOption rec_option;
    det_option.UseOrtBackend();
    rec_option.UseOrtBackend();

    const auto backend = m_gpu_selector ? m_gpu_selector->backend() : InferenceBackend::Auto;
    const auto device_id = m_gpu_selector ? m_gpu_selector->resolve_device_id() : std::nullopt;

#ifdef MAA_WITH_OPENVINO
    if (!force_cpu && backend == InferenceBackend::OpenVINO) {
        if (create_openvino_pipeline()) {
            return true;
        }
        Log.error(__FUNCTION__, "| OpenVINO OCR init failed, falling back to FastDeploy CPU");
    }
#endif

    if (!force_cpu) {
        if (backend == InferenceBackend::WebGPU) {
            if (!device_id) {
                Log.error(__FUNCTION__, "| failed to resolve the configured WebGPU device, falling back to CPU");
            }
            else if (auto webgpu_options = make_webgpu_provider_options(*m_gpu_selector); webgpu_options) {
                m_gpu_active = true;
                // FastDeploy 建 OrtSessionOptions 时只给 deviceId 不满足 WebGPU EP 的要求（见 WebGpuDevice.h），
                // 这里接管会话配置，把 Dawn 的 instance/device 一并交给 WebGPU EP。
                // UseWebGPU 的参数只是适配器序号；真正的 ORT context id 在 webgpu_options 里。
                m_impl->webgpu_options = std::move(*webgpu_options);
                det_option.ort_option.configure_session_callback = &configure_webgpu_session;
                det_option.ort_option.configure_session_callback_data = &m_impl->webgpu_options;
                rec_option.ort_option.configure_session_callback = &configure_webgpu_session;
                rec_option.ort_option.configure_session_callback_data = &m_impl->webgpu_options;
                det_option.UseWebGPU(*device_id);
                rec_option.UseWebGPU(*device_id);
                Log.info(__FUNCTION__, "| FastDeploy WebGPU mode with device", *device_id);
            }
            else {
                Log.error(__FUNCTION__, "| failed to create the WebGPU device, falling back to CPU");
            }
        }
#ifdef _WIN32
        else if (backend != InferenceBackend::OpenVINO && device_id) {
            m_gpu_active = true;
            det_option.UseDirectML(*device_id);
            rec_option.UseDirectML(*device_id);
            Log.info(__FUNCTION__, "| FastDeploy DirectML mode with device", *device_id);
        }
#endif
        else if (backend == InferenceBackend::DirectML) {
#ifdef _WIN32
            Log.error(__FUNCTION__, "| failed to resolve the configured DirectML device, falling back to CPU");
#else
            Log.error(__FUNCTION__, "| DirectML backend is only available on Windows, falling back to CPU");
#endif
        }
        else if (m_gpu_selector && !device_id) {
            Log.error(__FUNCTION__, "| failed to resolve the configured GPU, falling back to CPU");
        }
    }

    if (!m_gpu_active) {
        // macOS 上 CoreML 的 det/rec 结果不对（疑似丢精度），继续禁用：
        // https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h
        // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE
        // det_option.UseCoreML(0x004);
        // rec_option.UseCoreML(0x004);
        det_option.UseCpu();
        rec_option.UseCpu();
        // CPU 模式下限制线程数，避免过高的 CPU 占用
        det_option.SetCpuThreadNum(cpu_threads);
        rec_option.SetCpuThreadNum(cpu_threads);
        Log.info(__FUNCTION__, "| FastDeploy CPU mode with", cpu_threads, "threads");
    }

    try {
        m_impl->det = std::make_unique<fastdeploy::vision::ocr::DBDetector>(
            platform::path_to_utf8_string(m_impl->det_model_path),
            std::string(),
            det_option,
            fastdeploy::ModelFormat::ONNX);

        // 针对 PP-OCRv6 检测模型调优后处理参数
        if (m_impl->det && m_impl->det->Initialized()) {
            m_impl->det->GetPostprocessor().SetDetDBThresh(0.2);
            m_impl->det->GetPostprocessor().SetDetDBBoxThresh(0.45);
            m_impl->det->GetPostprocessor().SetDetDBUnclipRatio(1.4);
            m_impl->det->GetPostprocessor().SetDetDBMaxCandidates(3000);
        }

        m_impl->rec = std::make_unique<fastdeploy::vision::ocr::Recognizer>(
            platform::path_to_utf8_string(m_impl->rec_model_path),
            std::string(),
            platform::path_to_utf8_string(m_impl->rec_label_path),
            rec_option,
            fastdeploy::ModelFormat::ONNX);

        if (m_impl->det && m_impl->rec) {
            m_impl->ocr = std::make_unique<fastdeploy::pipeline::PPOCRv6>(m_impl->det.get(), m_impl->rec.get());
        }
    }
    catch (const std::exception& e) {
        Log.error(__FUNCTION__, "| failed to create OCR pipeline:", e.what());
        release_models();
        return false;
    }
    catch (...) {
        Log.error(__FUNCTION__, "| failed to create OCR pipeline: unknown exception");
        release_models();
        return false;
    }

    const bool det_inited = m_impl->det && m_impl->det->Initialized();
    const bool rec_inited = m_impl->rec && m_impl->rec->Initialized();
    const bool ocr_inited = m_impl->ocr && m_impl->ocr->Initialized();
    Log.info("det", det_inited, "rec", rec_inited, "ocr", ocr_inited);

    if (!pipeline_ready()) {
        Log.error(__FUNCTION__, "| OCR pipeline is not initialized");
        release_models();
        return false;
    }

    if (m_gpu_active && backend == InferenceBackend::WebGPU && !warmup()) {
        Log.error(__FUNCTION__, "| WebGPU warmup failed, falling back to CPU");
        release_models();
        return false;
    }

    return true;
}

bool OcrPack::warmup()
{
    LogTraceFunction;

    // WebGPU compiles its shaders on the first inference of a session (a few
    // hundred ms per model). One dummy inference per model moves that cost into
    // the loading phase instead of the first recognition.
    try {
        const cv::Mat det_dummy = cv::Mat::zeros(960, 960, CV_8UC3);
        std::vector<std::array<int, 8>> boxes;
        if (!m_impl->det->Predict(det_dummy, &boxes)) {
            Log.warn(__FUNCTION__, "| det warmup failed");
            return false;
        }

        const cv::Mat rec_dummy = cv::Mat::zeros(48, 320, CV_8UC3);
        std::string text;
        float score = 0.0F;
        if (!m_impl->rec->Predict(rec_dummy, &text, &score)) {
            Log.warn(__FUNCTION__, "| rec warmup failed");
            return false;
        }
    }
    catch (const std::exception& e) {
        Log.warn(__FUNCTION__, "| warmup failed:", e.what());
        return false;
    }
    catch (...) {
        Log.warn(__FUNCTION__, "| warmup failed: unknown exception");
        return false;
    }

    return true;
}

#ifdef MAA_WITH_OPENVINO
void OcrPack::release_openvino()
{
    m_ov_active = false;
    m_impl->ov_det_ready = false;
    m_impl->ov_det_req = ov::InferRequest();
    m_impl->ov_det_model = ov::CompiledModel();
    m_impl->ov_rec_buckets.clear();
    m_impl->ov_long_bucket_failed = false;
    m_impl->ov_det_in.clear();
    m_impl->ov_rec_in.clear();
}

bool OcrPack::create_openvino_pipeline()
{
#if defined(_DEBUG)
    // OpenVINO ships a release CRT only. Passing ov::Core/Tensor objects across
    // a /MDd MaaCore and /MD openvino.dll corrupts the heap, so debug builds
    // keep OCR on FastDeploy. ONNX sessions still use the OpenVINO EP via ORT.
    LogWarn << __FUNCTION__ << "native OpenVINO OCR is disabled in debug builds; using FastDeploy CPU";
    return false;
#else
    LogTraceFunction;

    if (!m_gpu_selector) {
        return false;
    }
    if (m_impl->det_model_path.empty() || m_impl->rec_model_path.empty() || m_impl->rec_label_path.empty()) {
        Log.error(__FUNCTION__, "| OCR model paths are incomplete");
        return false;
    }

    if (m_impl->ov_keys.empty()) {
        m_impl->ov_keys = ocr_ov::load_keys(m_impl->rec_label_path);
        if (m_impl->ov_keys.empty()) {
            Log.error(__FUNCTION__, "| failed to load keys from", m_impl->rec_label_path);
            return false;
        }
    }

    m_impl->ov_props.clear();
    m_impl->ov_props.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    const auto compile_all = [this](const std::string& device) {
        const ov::Shape det_shape { 1, 3, ocr_ov::DetHeight, ocr_ov::DetWidth };
        auto det_model = ocr_ov::read_static_model(ocr_ov::shared_core(), m_impl->det_model_path, det_shape);
        m_impl->ov_det_model = ocr_ov::shared_core().compile_model(det_model, device, m_impl->ov_props);
        m_impl->ov_det_req = m_impl->ov_det_model.create_infer_request();
        m_impl->ov_det_ready = true;
        LogInfo << "OpenVINO det execution devices" << ocr_ov::device_summary(m_impl->ov_det_model, device);

        const ov::Shape rec_shape { 1, 3, ocr_ov::RecHeight, ocr_ov::RecWidthShort };
        auto rec_model = ocr_ov::read_static_model(ocr_ov::shared_core(), m_impl->rec_model_path, rec_shape);
        auto compiled = ocr_ov::shared_core().compile_model(rec_model, device, m_impl->ov_props);
        auto request = compiled.create_infer_request();
        LogInfo << "OpenVINO rec execution devices" << ocr_ov::device_summary(compiled, device);
        m_impl->ov_rec_buckets.erase(ocr_ov::RecWidthShort);
        m_impl->ov_rec_buckets.emplace(
            ocr_ov::RecWidthShort,
            Impl::RecBucket { std::move(compiled), std::move(request) });
    };

    const auto warmup = [this] {
        float scale = 1.0F;
        const cv::Mat det_dummy = cv::Mat::zeros(ocr_ov::DetHeight, ocr_ov::DetWidth, CV_8UC3);
        ocr_ov::fill_det_input(det_dummy, m_impl->ov_det_in, scale);
        ov::Tensor det_tensor(
            ov::element::f32,
            ov::Shape { 1, 3, ocr_ov::DetHeight, ocr_ov::DetWidth },
            m_impl->ov_det_in.data());
        m_impl->ov_det_req.set_input_tensor(det_tensor);
        m_impl->ov_det_req.infer();

        const cv::Mat rec_dummy = cv::Mat::zeros(ocr_ov::RecHeight, ocr_ov::RecWidthShort, CV_8UC3);
        ocr_ov::fill_rec_input(rec_dummy, m_impl->ov_rec_in, ocr_ov::RecWidthShort);
        ov::Tensor rec_tensor(
            ov::element::f32,
            ov::Shape { 1, 3, ocr_ov::RecHeight, ocr_ov::RecWidthShort },
            m_impl->ov_rec_in.data());
        auto& request = m_impl->ov_rec_buckets.at(ocr_ov::RecWidthShort).request;
        request.set_input_tensor(rec_tensor);
        request.infer();
    };

    const std::string requested_device = m_gpu_selector->openvino_device() == OpenVINODevice::NPU ? "NPU" : "CPU";

    try {
        compile_all(requested_device);
        warmup();
        m_impl->ov_device = requested_device;
        m_ov_active = true;
        LogInfo << __FUNCTION__ << "OpenVINO OCR ready on" << requested_device << "det bucket" << ocr_ov::DetWidth
                << "x" << ocr_ov::DetHeight << "rec bucket" << ocr_ov::RecWidthShort << "x" << ocr_ov::RecHeight
                << "keys" << m_impl->ov_keys.size();
        return true;
    }
    catch (const std::exception& e) {
        LogWarn << __FUNCTION__ << "OpenVINO" << requested_device << "compile failed:" << e.what()
                << "falling back to OpenVINO CPU";
    }

    release_openvino();
    try {
        compile_all("CPU");
        warmup();
        m_impl->ov_device = "CPU";
        m_ov_active = true;
        LogInfo << __FUNCTION__ << "OpenVINO OCR ready on CPU after fallback";
        return true;
    }
    catch (const std::exception& e) {
        LogError << __FUNCTION__ << "OpenVINO CPU fallback failed:" << e.what();
        release_openvino();
        return false;
    }
#endif
}

OcrPack::ResultsVec
    OcrPack::recognize_openvino(const cv::Mat& image, bool without_det, const std::optional<Rect>& base_roi)
{
    if (image.empty()) {
        return {};
    }

    const auto start_time = std::chrono::steady_clock::now();
    ResultsVec raw_results;

    const auto ensure_bucket = [this](int width) -> bool {
        if (m_impl->ov_rec_buckets.contains(width)) {
            return true;
        }
        if (width == ocr_ov::RecWidthLong && m_impl->ov_long_bucket_failed) {
            return false;
        }
        try {
            const ov::Shape shape { 1, 3, ocr_ov::RecHeight, static_cast<std::size_t>(width) };
            auto model = ocr_ov::read_static_model(ocr_ov::shared_core(), m_impl->rec_model_path, shape);
            auto compiled = ocr_ov::shared_core().compile_model(model, m_impl->ov_device, m_impl->ov_props);
            auto request = compiled.create_infer_request();
            LogInfo << "OpenVINO extra rec bucket" << width << "execution devices"
                    << ocr_ov::device_summary(compiled, m_impl->ov_device);
            m_impl->ov_rec_buckets.emplace(width, Impl::RecBucket { std::move(compiled), std::move(request) });
            return true;
        }
        catch (const std::exception& e) {
            LogWarn << __FUNCTION__ << "failed to compile rec bucket" << width << ":" << e.what();
            if (width == ocr_ov::RecWidthLong) {
                m_impl->ov_long_bucket_failed = true;
            }
            return false;
        }
    };

    const auto pick_bucket = [&](const cv::Mat& crop) {
        const int scaled_width = ocr_ov::rec_scaled_width(crop);
        if (scaled_width <= ocr_ov::RecWidthShort) {
            return ocr_ov::RecWidthShort;
        }
        if (ensure_bucket(ocr_ov::RecWidthLong)) {
            return ocr_ov::RecWidthLong;
        }
        LogTrace << __FUNCTION__ << "line width" << scaled_width << "exceeds" << ocr_ov::RecWidthShort
                 << "and the long bucket is unavailable; right side clipped";
        return ocr_ov::RecWidthShort;
    };

    const auto run_rec = [this](int bucket, float& out_score) -> std::string {
        const auto bucket_iter = m_impl->ov_rec_buckets.find(bucket);
        if (bucket_iter == m_impl->ov_rec_buckets.end()) {
            out_score = 0.0F;
            return {};
        }
        ov::Tensor input(
            ov::element::f32,
            ov::Shape { 1, 3, ocr_ov::RecHeight, static_cast<std::size_t>(bucket) },
            m_impl->ov_rec_in.data());
        {
            const std::lock_guard lock(ocr_ov::npu_mutex());
            bucket_iter->second.request.set_input_tensor(input);
            bucket_iter->second.request.infer();
        }

        const ov::Tensor output = bucket_iter->second.request.get_output_tensor(0);
        const ov::Shape shape = output.get_shape();
        const int steps = static_cast<int>(shape[shape.size() - 2]);
        const int classes = static_cast<int>(shape[shape.size() - 1]);
        return ocr_ov::ctc_decode(output.data<const float>(), steps, classes, m_impl->ov_keys, out_score);
    };

    try {
        if (without_det) {
            const int bucket = pick_bucket(image);
            ocr_ov::fill_rec_input(image, m_impl->ov_rec_in, bucket);
            float score = 0.0F;
            std::string text = run_rec(bucket, score);
            raw_results.emplace_back(Rect(0, 0, image.cols, image.rows), score, std::move(text));
        }
        else {
            float det_scale = 1.0F;
            ocr_ov::fill_det_input(image, m_impl->ov_det_in, det_scale);
            ov::Tensor det_tensor(
                ov::element::f32,
                ov::Shape { 1, 3, ocr_ov::DetHeight, ocr_ov::DetWidth },
                m_impl->ov_det_in.data());
            {
                const std::lock_guard lock(ocr_ov::npu_mutex());
                m_impl->ov_det_req.set_input_tensor(det_tensor);
                m_impl->ov_det_req.infer();
            }

            const ov::Tensor det_output = m_impl->ov_det_req.get_output_tensor(0);
            const ov::Shape det_shape = det_output.get_shape();
            const int prob_h = static_cast<int>(det_shape[det_shape.size() - 2]);
            const int prob_w = static_cast<int>(det_shape[det_shape.size() - 1]);

            cv::Mat prob(prob_h, prob_w, CV_32FC1, const_cast<float*>(det_output.data<const float>()));
            cv::Mat bitmap;
            cv::threshold(prob, bitmap, ocr_ov::DetBinThreshold, 255.0, cv::THRESH_BINARY);
            bitmap.convertTo(bitmap, CV_8UC1);

            const auto boxes = db_postprocess::boxes_from_bitmap(
                prob,
                bitmap,
                ocr_ov::BoxThreshold,
                ocr_ov::UnclipRatio,
                ocr_ov::MaxCandidates);

            for (auto quad : boxes) {
                for (auto& point : quad) {
                    point.x /= det_scale;
                    point.y /= det_scale;
                }

                const int left = std::clamp(
                    static_cast<int>(std::min({ quad[0].x, quad[1].x, quad[2].x, quad[3].x })),
                    0,
                    image.cols - 1);
                const int right = std::clamp(
                    static_cast<int>(std::max({ quad[0].x, quad[1].x, quad[2].x, quad[3].x })),
                    0,
                    image.cols - 1);
                const int top = std::clamp(
                    static_cast<int>(std::min({ quad[0].y, quad[1].y, quad[2].y, quad[3].y })),
                    0,
                    image.rows - 1);
                const int bottom = std::clamp(
                    static_cast<int>(std::max({ quad[0].y, quad[1].y, quad[2].y, quad[3].y })),
                    0,
                    image.rows - 1);
                if (right <= left || bottom <= top) {
                    continue;
                }

                cv::Mat crop = db_postprocess::get_rotate_crop_image(image, quad);
                if (crop.empty() || crop.rows < 4 || crop.cols < 4) {
                    continue;
                }

                const int bucket = pick_bucket(crop);
                ocr_ov::fill_rec_input(crop, m_impl->ov_rec_in, bucket);
                float score = 0.0F;
                std::string text = run_rec(bucket, score);
                if (text.empty()) {
                    continue;
                }
                raw_results.emplace_back(Rect(left, top, right - left, bottom - top), score, std::move(text));
            }
        }
    }
    catch (const std::exception& e) {
        LogError << __FUNCTION__ << "OpenVINO inference failed:" << e.what();
    }

    const auto costs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
    std::string class_type = utils::demangle(typeid(*this).name());
    if (!base_roi) {
        Log.trace(
            class_type,
            raw_results,
            without_det ? "by OCR Rec (OpenVINO)" : "by OCR Pipeline (OpenVINO)",
            ", cost",
            costs,
            "ms");
    }
    else {
        std::string output = "[";
        for (std::size_t i = 0; i < raw_results.size(); ++i) {
            if (i != 0) {
                output += ", ";
            }
            output += std::format(
                "{{ text: {}, rect: [ {} ({}), {} ({}), {}, {} ], score: {:.6f} }}",
                raw_results[i].text,
                raw_results[i].rect.x,
                raw_results[i].rect.x + base_roi->x,
                raw_results[i].rect.y,
                raw_results[i].rect.y + base_roi->y,
                raw_results[i].rect.width,
                raw_results[i].rect.height,
                raw_results[i].score);
        }
        output += "]";
        Log.trace(
            class_type,
            output,
            without_det ? "by OCR Rec (OpenVINO)" : "by OCR Pipeline (OpenVINO)",
            ", cost",
            costs,
            "ms");
    }
    return raw_results;
}
#endif

} // namespace asst

#if defined(MAA_WITH_OPENVINO) && defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif // __ANDROID__
