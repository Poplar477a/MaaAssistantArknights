#include "OnnxSessions.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <future>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "Utils/Logger.hpp"
#include "Utils/Platform.hpp"
#include "Utils/WorkingDir.hpp"
#include "WebGpuDevice.h"

#if __has_include(<onnxruntime/dml_provider_factory.h>)
#define WITH_DML
#include <onnxruntime/dml_provider_factory.h>
#endif

#if defined(__APPLE__) && __has_include(<onnxruntime/coreml_provider_factory.h>)
#define WITH_COREML
#include <onnxruntime/coreml_provider_factory.h>
#endif

namespace asst::onnx_session_detail
{
// Destroying an ORT session on the OpenVINO NPU can crash inside the plugin.
// Keep abandoned sessions alive for the process lifetime instead.
void abandon_ort_session(std::string name, std::shared_ptr<Ort::Session> session)
{
    static auto* abandoned = new std::vector<std::pair<std::string, std::shared_ptr<Ort::Session>>>();
    static std::mutex abandoned_mutex;
    const std::lock_guard lock(abandoned_mutex);
    if (abandoned->size() >= 128) {
        // 反复热重载会累积已泄漏的 NPU 会话；提示用户重启而非无限增长。
        static std::atomic<bool> warned { false };
        if (!warned.exchange(true)) {
            LogWarn << "too many abandoned ONNX sessions; restart MAA to release NPU memory";
        }
    }
    abandoned->emplace_back(std::move(name), std::move(session));
}
}

bool asst::OnnxSessions::load(const std::filesystem::path& path)
{
    LogTraceFunction;
    LogInfo << "record path" << path.lexically_relative(UserDir.get());

    std::string name = utils::path_to_utf8_string(path.stem());
    std::lock_guard lock(m_mutex);
    if (auto iter = m_model_paths.find(name); iter == m_model_paths.end() || iter->second != path) {
        m_model_paths.insert_or_assign(name, path);
        // 引用归零前不能销毁会话，持有者（如地图感知）还握着裸引用；挂起待 release 时销毁
        if (m_session_users.contains(name)) {
            m_pending_reload.insert(name);
        }
        else if (auto session = m_sessions.find(name); session != m_sessions.end()) {
            // Callers may still hold the reference returned by get(); keep the
            // object alive instead of destroying it with the cache entry.
            onnx_session_detail::abandon_ort_session(session->first, std::move(session->second));
            m_sessions.erase(session);
        }
    }

    if (gpu_enabled && m_gpu_selector &&
        (m_gpu_selector->backend() == InferenceBackend::WebGPU ||
         m_gpu_selector->backend() == InferenceBackend::OpenVINO) &&
        !m_sessions.contains(name)) {
        warmup_locked(name);
    }

    return true;
}

void asst::OnnxSessions::warmup_locked(const std::string& name)
{
    // WebGPU 的 shader 编译发生在 session 第一次推理上（实测 operators_det ~330ms、
    // skill_ready_cls ~160ms、deploy_direction_cls ~157ms）。这些模型的输入 rank 在
    // onnx 里没写，形状只能在这里给出，取值与各调用方喂进去的一致。
    static const std::unordered_map<std::string, std::pair<int, int>> kInputSizes = {
        { "operators_det", { 640, 640 } },
        { "skill_ready_cls", { 64, 64 } },
        { "deploy_direction_cls", { 96, 96 } },
        { "BlackFlow_corridor_net", { 40, 160 } },
    };

    const auto size = kInputSizes.find(name);
    if (size == kInputSizes.end()) {
        LogDebug << __FUNCTION__ << "| no warmup size recorded for" << name;
        return;
    }

    const std::array<int64_t, 4> shape { 1, 3, size->second.first, size->second.second };
    std::vector<float> data(static_cast<std::size_t>(shape[0] * shape[1] * shape[2] * shape[3]), 0.0F);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), shape.data(), shape.size());

    const auto run = [&](Ort::Session& session) {
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[] = { input_name.get() };
        const char* output_names[] = { output_name.get() };
        auto npu_lock = OnnxSessions::acquire_npu_inference_lock();
        session.Run(Ort::RunOptions { nullptr }, input_names, &input, 1, output_names, 1);
    };

    try {
        run(get_or_create(name));
        LogInfo << __FUNCTION__ << "| warmed up" << name;
        return;
    }
    catch (const std::exception& e) {
        LogWarn << __FUNCTION__ << "| warmup failed" << name << e.what();
    }

    // 建 session 成功不代表设备能跑。预热是第一次真正推理，失败时拆掉 GPU 会话再按 CPU 建一次，
    // 否则这次会话会留在缓存里，后面的推理继续撞同一块坏设备。
    if (!gpu_enabled) {
        return;
    }

    LogError << __FUNCTION__ << "accelerator warmup failed for" << name << "; falling back to CPU mode";
    const bool leak_npu_sessions = m_npu_inference_enabled.load(std::memory_order_relaxed);
    if (leak_npu_sessions) {
        // Callers using get() hold raw session references without registering
        // in m_session_users. Leak all cached sessions so those references stay
        // valid while the cache is rebuilt on CPU.
        for (auto& [session_name, session] : m_sessions) {
            onnx_session_detail::abandon_ort_session(session_name, std::move(session));
        }
        m_sessions.clear();
        m_pending_reload.clear();
    }
    else {
        if (auto iter = m_sessions.find(name); iter != m_sessions.end()) {
            onnx_session_detail::abandon_ort_session(iter->first, std::move(iter->second));
            m_sessions.erase(iter);
        }
        for (auto iter = m_sessions.begin(); iter != m_sessions.end();) {
            if (m_session_users.contains(iter->first)) {
                m_pending_reload.insert(iter->first);
                ++iter;
            }
            else {
                onnx_session_detail::abandon_ort_session(iter->first, std::move(iter->second));
                iter = m_sessions.erase(iter);
            }
        }
    }
    reset_session_options();
    m_gpu_selector = std::nullopt;
    gpu_enabled = false;
    gpu_options_initialized = false;
    // Abandoned NPU sessions may still be referenced by callers, so keep the NPU
    // lock enabled; it only serializes inference, which is safe for CPU sessions.
    m_npu_inference_enabled.store(leak_npu_sessions || !m_sessions.empty(), std::memory_order_relaxed);

    try {
        const auto created =
            m_sessions.emplace(name, std::make_shared<Ort::Session>(m_env, m_model_paths.at(name).c_str(), m_options));
        run(*created.first->second);
        LogInfo << __FUNCTION__ << "| warmed up" << name << "on CPU";
    }
    catch (const Ort::Exception& ex) {
        LogError << __FUNCTION__ << "Failed to recreate" << name << "on CPU:" << ex.what();
        if (auto iter = m_sessions.find(name); iter != m_sessions.end()) {
            onnx_session_detail::abandon_ort_session(iter->first, std::move(iter->second));
            m_sessions.erase(iter);
        }
        throw;
    }
    catch (const std::exception& e) {
        LogWarn << __FUNCTION__ << "| CPU warmup failed" << name << e.what();
    }
}

Ort::Session& asst::OnnxSessions::get_or_create(const std::string& name)
{
    if (m_sessions.contains(name)) {
        return *m_sessions.at(name);
    }

    if (gpu_enabled && !gpu_options_initialized && !initialize_gpu_options()) {
        LogError << __FUNCTION__ << "Failed to initialize configured GPU; falling back to CPU mode";
        use_cpu_locked();
    }

    LogInfo << __FUNCTION__ << "lazy load" << name;
    // NPU 会话编译与原生 OCR 共用插件，必须与 OCR 串行。首次创建时
    // m_npu_inference_enabled 尚未置位，需要直接看当前 selector。
    const bool npu_selector = gpu_enabled && m_gpu_selector &&
                              m_gpu_selector->backend() == InferenceBackend::OpenVINO &&
                              m_gpu_selector->openvino_device() == OpenVINODevice::NPU;
    std::unique_lock<std::timed_mutex> npu_guard;
    if (npu_selector || m_npu_inference_enabled.load(std::memory_order_relaxed)) {
        npu_guard = std::unique_lock<std::timed_mutex>(openvino_npu_mutex());
    }
    try {
        m_sessions.emplace(name, std::make_shared<Ort::Session>(m_env, m_model_paths.at(name).c_str(), m_options));
    }
    catch (const Ort::Exception& ex) {
        // GPU 后端的初始化错误（例如 WebGPU EP）可能到建 session 时才暴露，退回 CPU 重试一次
        LogError << __FUNCTION__ << "Failed to create" << name << "session:" << ex.what();
        if (!gpu_enabled) {
            throw;
        }

        LogError << __FUNCTION__ << "Falling back to CPU mode";
        reset_session_options();
        m_gpu_selector = std::nullopt;
        gpu_enabled = false;
        gpu_options_initialized = false;
        // 与 warmup 回退保持一致：放弃全部已缓存会话，避免部分会话留在加速器上。
        // 不销毁：持有引用的调用方可能仍在运行。不清除 NPU 标志原因相同。
        for (auto iter = m_sessions.begin(); iter != m_sessions.end();) {
            onnx_session_detail::abandon_ort_session(iter->first, std::move(iter->second));
            iter = m_sessions.erase(iter);
        }
        m_sessions.emplace(name, std::make_shared<Ort::Session>(m_env, m_model_paths.at(name).c_str(), m_options));
    }

    return *m_sessions.at(name);
}

Ort::Session& asst::OnnxSessions::get(const std::string& name)
{
    std::lock_guard lock(m_mutex);
    return get_or_create(name);
}

Ort::Session& asst::OnnxSessions::acquire(const std::string& name)
{
    std::lock_guard lock(m_mutex);
    Ort::Session& session = get_or_create(name);
    ++m_session_users[name];
    return session;
}

void asst::OnnxSessions::release(const std::string& name)
{
    std::lock_guard lock(m_mutex);
    const auto found = m_session_users.find(name);
    if (found == m_session_users.end()) {
        LogError << __FUNCTION__ << "session was not acquired" << name;
        return;
    }
    if (--found->second == 0) {
        m_session_users.erase(found);
        if (m_pending_reload.erase(name) > 0) {
            // 持有期间模型路径变化过，最后一个引用释放后放弃旧会话，下次 acquire 按新路径重建。
            // 不销毁：其他调用方可能仍持有 get() 返回的引用。
            if (auto iter = m_sessions.find(name); iter != m_sessions.end()) {
                onnx_session_detail::abandon_ort_session(iter->first, std::move(iter->second));
                m_sessions.erase(iter);
            }
            LogInfo << __FUNCTION__ << "stale session destroyed after last release" << name;
        }
        // 不依据缓存内容清除 NPU 标志：在途引用可能仍在 NPU 上运行，串行锁必须保持生效。
        LogInfo << __FUNCTION__ << "released" << name;
    }
}

int asst::OnnxSessions::reset_session_options()
{
    m_options = Ort::SessionOptions();

    const int logical = std::max(1u, std::thread::hardware_concurrency());
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

    // 设置执行模式为顺序执行，减少线程竞争
    m_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    m_options.SetIntraOpNumThreads(cpu_threads);

    return cpu_threads;
}

bool asst::OnnxSessions::use_cpu()
{
    std::lock_guard lock(m_mutex);
    return use_cpu_locked();
}

bool asst::OnnxSessions::use_cpu_locked()
{
    if (!m_sessions.empty()) {
        return false;
    }

    const auto cpu_threads = reset_session_options();
    LogInfo << "CPU OCR enabled with" << cpu_threads << "threads";

    m_gpu_selector = std::nullopt;
    gpu_enabled = false;
    gpu_options_initialized = false;
    // 不在此清除 NPU 标志：已放弃的 NPU 会话可能仍被调用方持有并在运行。
    return true;
}

bool asst::OnnxSessions::use_gpu(GpuDeviceSelector selector)
{
    std::lock_guard lock(m_mutex);
    return use_gpu_locked(std::move(selector));
}

bool asst::OnnxSessions::use_gpu_locked(GpuDeviceSelector selector)
{
    if (gpu_enabled) {
        if (m_gpu_selector == selector) {
            return true;
        }

        LogError << __FUNCTION__ << "GPU OCR is already configured with a different device selector";
        return false;
    }
    if (!m_sessions.empty()) {
        LogError << __FUNCTION__ << "GPU OCR cannot be configured after ONNX sessions have been created";
        return false;
    }

    reset_session_options();
    m_gpu_selector = std::move(selector);
    gpu_enabled = true;
    gpu_options_initialized = false;
    return true;
}

bool asst::OnnxSessions::initialize_gpu_options()
{
    if (!m_gpu_selector) {
        return false;
    }

    const auto device_id = m_gpu_selector->resolve_device_id();
    if (!device_id) {
        return false;
    }

    const auto backend = m_gpu_selector->backend();
    // 不在此清除 NPU 标志：已放弃的 NPU 会话可能仍被在途调用使用。
    auto all_providers = Ort::GetAvailableProviders();
    bool support_cuda = false;
#ifdef WITH_DML
    bool support_dml = false;
#endif
#ifdef WITH_COREML
    bool support_coreml = false;
#endif
    bool support_webgpu = false;
    bool support_openvino = false;
    for (const auto& provider : all_providers) {
        if (provider == "CUDAExecutionProvider") {
            support_cuda = true;
        }
#ifdef WITH_DML
        else if (provider == "DmlExecutionProvider") {
            support_dml = true;
        }
#endif
#ifdef WITH_COREML
        else if (provider == "CoreMLExecutionProvider") {
            support_coreml = true;
        }
#endif
        else if (provider == "WebGpuExecutionProvider") {
            support_webgpu = true;
        }
        else if (provider == "OpenVINOExecutionProvider") {
            support_openvino = true;
        }
    }

    bool provider_configured = false;

    if (backend == InferenceBackend::WebGPU) {
        if (support_webgpu) {
            // SessionOptionsAppendExecutionProvider prefixes the key with
            // "ep.webgpuexecutionprovider." by itself, so a fully qualified key here would be ignored.
            // deviceId > 0 时 ORT 要求自带 WebGPU instance/device，由 make_webgpu_provider_options 一并给出。
            const auto ep_options = make_webgpu_provider_options(*m_gpu_selector);
            if (!ep_options) {
                LogError << "Failed to create the WebGPU device for device" << *device_id;
                return false;
            }

            try {
                m_options.AppendExecutionProvider("WebGPU", *ep_options);
                provider_configured = true;
                LogInfo << "WebGPU execution provider enabled for device" << *device_id;
            }
            catch (const Ort::Exception& ex) {
                LogError << "Failed to append WebGPU execution provider:" << ex.what();
                return false;
            }
        }
        else {
            LogError << "WebGPU execution provider requested but not available in this build";
            return false;
        }
    }
    else if (backend == InferenceBackend::OpenVINO) {
        if (support_openvino) {
            const bool npu = m_gpu_selector->openvino_device() == OpenVINODevice::NPU;
            std::string device_type = npu ? "NPU" : "CPU";
            if (*device_id > 0) {
                device_type += "." + std::to_string(*device_id);
            }

            std::unordered_map<std::string, std::string> ep_options {
                { "device_type", std::move(device_type) },
                { "precision", npu ? "FP16" : "FP32" },
            };
            // NPU only accepts static shapes. The flag rewrites the two dynamic
            // batch dimensions (skill_ready_cls / BlackFlow_corridor_net) to 1.
            if (npu) {
                ep_options.emplace("disable_dynamic_shapes", "true");
            }

            try {
                m_options.AppendExecutionProvider_OpenVINO_V2(ep_options);
                provider_configured = true;
                if (npu) {
                    // 只置不清：已放弃的 NPU 会话可能仍被调用方持有。
                    m_npu_inference_enabled.store(true, std::memory_order_relaxed);
                }
                LogInfo << "OpenVINO execution provider enabled on" << ep_options.at("device_type");
            }
            catch (const Ort::Exception& ex) {
                LogError << "Failed to append OpenVINO execution provider:" << ex.what();
                return false;
            }
        }
        else {
            LogError << "OpenVINO execution provider requested but not available in this build";
            return false;
        }
    }
    else if (backend == InferenceBackend::DirectML) {
#ifdef WITH_DML
        if (support_dml) {
            if (!Ort::Status(OrtSessionOptionsAppendExecutionProvider_DML(m_options, *device_id)).IsOK()) {
                LogError << "Failed to append DirectML execution provider for device" << *device_id;
                return false;
            }
            provider_configured = true;
            LogInfo << "DirectML execution provider enabled for device" << *device_id;
        }
        else
#endif
        {
            LogError << "DirectML execution provider requested but not available in this build";
            return false;
        }
    }
    else {
        // Auto: 优先使用最成熟稳定的后端，跳过实验性的 WebGPU
        if (support_cuda) {
            OrtCUDAProviderOptions cuda_options {};
            cuda_options.device_id = *device_id;
            m_options.AppendExecutionProvider_CUDA(cuda_options);
            provider_configured = true;
            LogInfo << "CUDA execution provider enabled for device" << *device_id;
        }
#ifdef WITH_DML
        else if (support_dml) {
            if (!Ort::Status(OrtSessionOptionsAppendExecutionProvider_DML(m_options, *device_id)).IsOK()) {
                return false;
            }
            provider_configured = true;
            LogInfo << "DirectML execution provider enabled for device" << *device_id;
        }
#endif
#ifdef WITH_COREML
        else if (support_coreml) {
            if (!Ort::Status(OrtSessionOptionsAppendExecutionProvider_CoreML((OrtSessionOptions*)m_options, 0))
                     .IsOK()) {
                return false;
            }
            provider_configured = true;
            LogInfo << "CoreML execution provider enabled";
        }
#endif
    }

    if (!provider_configured) {
        LogError << "No GPU execution provider available";
        return false;
    }

    gpu_options_initialized = true;
    return true;
}

static void run_probe_session(Ort::Session& session)
{
    std::array<float, 1 * 3 * 96 * 96> zeros {};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> shape { 1, 3, 96, 96 };
    auto input = Ort::Value::CreateTensor<float>(memory_info, zeros.data(), zeros.size(), shape.data(), shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = { input_name.get() };
    const char* output_names[] = { output_name.get() };
    session.Run(Ort::RunOptions { nullptr }, input_names, &input, 1, output_names, 1);
}

bool asst::OnnxSessions::probe_openvino(OpenVINODevice device)
{
    // SessionOptions needs a registered default logger; the singleton's Env is
    // not reachable from this static query, so keep one probe-only Env alive.
    static Ort::Env* probe_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "openvino-probe");
    static std::mutex probe_session_mutex;
    static Ort::Session* npu_probe_session = nullptr;

    const bool npu = device == OpenVINODevice::NPU;

    using namespace asst::utils::path_literals;
    const auto model_path = UserDir.get() / "resource"_p / "onnx"_p / "deploy_direction_cls.onnx"_p;

    // Provider enumeration and EP registration can also block on a stalled
    // driver, so run all of the potentially blocking probe work in the timed
    // worker instead of only the session creation/inference.
    auto future = std::async(std::launch::async, [npu, model_path]() -> bool {
        try {
            const std::vector<std::string> all_providers = Ort::GetAvailableProviders();
            if (std::ranges::find(all_providers, "OpenVINOExecutionProvider") == all_providers.end()) {
                LogInfo << __FUNCTION__ << "| OpenVINO execution provider unavailable";
                return false;
            }

            if (!std::filesystem::exists(model_path)) {
                // The resource may live elsewhere (or not be installed yet). Don't
                // turn that into "device unavailable"; fall back to registration.
                LogWarn << __FUNCTION__ << "| probe model is missing:" << model_path << "EP registration only";
                return true;
            }

            Ort::SessionOptions options;
            std::unordered_map<std::string, std::string> ep_options {
                { "device_type", npu ? "NPU" : "CPU" },
            };
            if (npu) {
                ep_options.emplace("disable_dynamic_shapes", "true");
            }
            options.AppendExecutionProvider_OpenVINO_V2(ep_options);

            if (npu) {
                // Do not probe while NPU inference or OCR is active: take the
                // shared lock opportunistically instead of racing the plugin.
                std::unique_lock npu_lock(openvino_npu_mutex(), std::try_to_lock);
                if (!npu_lock.owns_lock()) {
                    // NPU 正在被使用，视为可用，避免误标不可用。
                    LogInfo << __FUNCTION__ << "| NPU busy, assuming available";
                    return true;
                }
                const std::lock_guard session_lock(probe_session_mutex);
                if (npu_probe_session == nullptr) {
                    // Keep the probe session alive: destroying an NPU ORT
                    // session can crash inside the plugin.
                    npu_probe_session = new Ort::Session(*probe_env, model_path.c_str(), options);
                }
                run_probe_session(*npu_probe_session);
            }
            else {
                Ort::Session session(*probe_env, model_path.c_str(), options);
                run_probe_session(session);
            }
            // ORT does not expose per-node device assignment. A successful run
            // through the OpenVINO EP is the strongest check available here;
            // the native OCR path logs the real EXECUTION_DEVICES.
            return true;
        }
        catch (const std::exception& e) {
            LogWarn << __FUNCTION__ << "| OpenVINO" << (npu ? "NPU" : "CPU") << "unavailable:" << e.what();
            return false;
        }
    });

    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        // Leak the future so its destructor does not block on a stuck probe.
        (void)new std::future<bool>(std::move(future));
        LogWarn << __FUNCTION__ << "| OpenVINO" << (npu ? "NPU" : "CPU") << "probe timed out";
        return false;
    }
    if (!future.get()) {
        return false;
    }

    LogInfo << __FUNCTION__ << "| OpenVINO" << (npu ? "NPU" : "CPU") << "available";
    return true;
}

bool asst::OnnxSessions::npu_inference_enabled() noexcept
{
    return get_instance().m_npu_inference_enabled.load(std::memory_order_relaxed);
}

std::unique_lock<std::timed_mutex> asst::OnnxSessions::acquire_npu_inference_lock()
{
    if (!npu_inference_enabled()) {
        return {};
    }
    // 必须真正持有锁才允许推理；等待而不是超时后无锁运行，避免与 OCR/NPU
    // 插件并发。驱动卡死的场景由探针的超时与重启策略兜底。
    return std::unique_lock<std::timed_mutex>(openvino_npu_mutex());
}

asst::OnnxSessions::~OnnxSessions()
{
    // FIXME: intentionally leak ort objects to avoid crash (double free?)
    // https://github.com/microsoft/onnxruntime/issues/15174
    auto leak_sessions = new decltype(m_sessions);
    *leak_sessions = std::move(m_sessions);

    auto leak_options = new Ort::SessionOptions(nullptr);
    *leak_options = std::move(m_options);
}
