const chooseFileButton = document.querySelector("#choose-file");
const runVideoButton = document.querySelector("#run-video");
const stopVideoButton = document.querySelector("#stop-video");
const exportVideoButton = document.querySelector("#export-video");
const startWebcamButton = document.querySelector("#start-webcam");
const imageFileInput = document.querySelector("#image-file-input");
const previewImage = document.querySelector("#preview-image");
const previewVideo = document.querySelector("#preview-video");
const previewCanvas = document.querySelector("canvas.media-layer");
const previewEmpty = document.querySelector("#preview-empty");
const previewEmptyTitle = previewEmpty?.querySelector("strong");
const previewEmptyDescription = previewEmpty?.querySelector("span:last-child");
const fileFeedback = document.querySelector("#file-feedback");
const mediaStatus = document.querySelector(".media-status");
const infoFilename = document.querySelector("#info-filename");
const infoResolution = document.querySelector("#info-resolution");
const infoDuration = document.querySelector("#info-duration");
const infoFormat = document.querySelector("#info-format");
const resultsList = document.querySelector("#results-list");
const resultsNote = document.querySelector("#results-note");
const appStateLabel = document.querySelector("#app-state-label");
const processingProgress = document.querySelector("#processing-progress");
const progressValue = document.querySelector("#progress-value");

const simulatedDetectionData = {
  frame_id: 125,
  detections: [
    { track_id: 7, class_name: "Nhựa", confidence: 0.92, bbox: [120, 80, 310, 260] },
    { track_id: 8, class_name: "Giấy", confidence: 0.85, bbox: [365, 115, 180, 210] },
  ],
};

let recordedChunks = [];






const videoMimeByExtension = {
  mp4: "video/mp4",
  avi: "video/x-msvideo",
  webm: "video/webm",
};

let previewObjectUrl = null;
let webcamStream = null;
let validationRequest = 0;
let activeMediaKind = null;
let appState = "Sẵn sàng";
let progressAnimationFrame = null;
let videoFrameAnimationFrame = null;
let videoFrameCallbackId = null;
let videoFrameLoopToken = 0;

if (
  chooseFileButton && startWebcamButton && runVideoButton && stopVideoButton && exportVideoButton &&
  imageFileInput && previewImage && previewVideo && previewCanvas && previewEmpty &&
  previewEmptyTitle && previewEmptyDescription && fileFeedback && mediaStatus &&
  infoFilename && infoResolution && infoDuration && infoFormat && resultsList && resultsNote &&
  appStateLabel && processingProgress && progressValue
) {
  chooseFileButton.disabled = false;
  runVideoButton.disabled = true;
  stopVideoButton.disabled = true;
  exportVideoButton.disabled = true;

  chooseFileButton.addEventListener("click", () => imageFileInput.click());
  startWebcamButton.addEventListener("click", () => void startWebcam());
  stopVideoButton.addEventListener("click", stopProcessingOrWebcam);
  runVideoButton.addEventListener("click", startProcessing);
  exportVideoButton.addEventListener("click", downloadRecordedVideo);
  previewVideo.addEventListener("ended", finishVideoProcessing);
  previewVideo.addEventListener("timeupdate", updateVideoProgress);
  document.addEventListener('workspace:leave', () => {
    if (appState === 'Đang xử lý') stopProcessingOrWebcam();
    pauseCanvasRecording();
    if (webcamStream || !activeMediaKind) {
      validationRequest += 1;
      clearPreview();
      setAppState('Sẵn sàng');
    }
  });
  document.addEventListener('workspace:reset', () => {
    validationRequest += 1;
    clearPreview();
    setAppState('Sẵn sàng');
    fileFeedback.hidden = true;
    resultsList.replaceChildren();
    document.querySelector('#render-fps').textContent = '—';
  });

  setProgress(0);
  setAppState("Sẵn sàng");

  imageFileInput.addEventListener("change", () => {
    const file = imageFileInput.files?.[0];
    imageFileInput.value = "";
    if (!file) return;

    validationRequest += 1;
    const thisRequest = validationRequest;
    clearPreview();
    setProgress(0);
    setAppState("Sẵn sàng");
    fileFeedback.hidden = true;
    fileFeedback.classList.remove("is-success");

    const extension = file.name.split(".").pop()?.toLowerCase();
    if (["png", "jpg", "jpeg"].includes(extension)) {
      validateImage(file, extension, thisRequest);
      return;
    }

    if (Object.hasOwn(videoMimeByExtension, extension)) {
      void loadVideoFile(file, extension, thisRequest);
      return;
    }

    showFileError("Định dạng tệp không được hỗ trợ.");
  });

  function validateImage(file, extension, requestId) {
    const mime = file.type.toLowerCase().split(";")[0];
    const validMime = !mime || mime === "application/octet-stream" || ["image/png", "image/jpeg"].includes(mime);
    if (!validMime) {
      showFileError("Lỗi tệp ảnh không hợp lệ");
      return;
    }

    try {
      previewObjectUrl = URL.createObjectURL(file);
      const image = new Image();

      image.onerror = () => {
        if (requestId === validationRequest) showFileError("Lỗi tệp ảnh không hợp lệ");
      };
      image.onload = () => {
        if (requestId !== validationRequest) return;
        if (image.naturalWidth === 0 || image.naturalHeight === 0) {
          showFileError("Lỗi tệp ảnh không hợp lệ");
          return;
        }

        previewImage.src = previewObjectUrl;
        previewImage.hidden = false;
        previewEmpty.hidden = true;
        activeMediaKind = "image";
        setAppState("Đã nạp");
        fileFeedback.textContent = `Đã tải ảnh hợp lệ: ${file.name}`;
        fileFeedback.classList.add("is-success");
        fileFeedback.hidden = false;
        infoFilename.textContent = file.name;
        infoFilename.title = file.name;
        infoResolution.textContent = `${image.naturalWidth} × ${image.naturalHeight}`;
        infoDuration.textContent = "—";
        infoFormat.textContent = extension.toUpperCase();
      };

      image.src = previewObjectUrl;
    } catch {
      if (requestId === validationRequest) showFileError("Lỗi tệp ảnh không hợp lệ");
    }
  }

  async function loadVideoFile(file, extension, requestId) {
    const declaredMime = file.type.toLowerCase().split(";")[0];
    const expectedMime = videoMimeByExtension[extension];
    const validMime = !declaredMime || declaredMime === "application/octet-stream" ||
      declaredMime === expectedMime || (extension === "avi" && declaredMime === "video/avi");

    if (!validMime) {
      showFileError("Định dạng video không được hỗ trợ.");
      return;
    }

    const mime = declaredMime && declaredMime !== "application/octet-stream" ? declaredMime : expectedMime;
    if (previewVideo.canPlayType(mime) === "") {
      showFileError("Định dạng video không được hỗ trợ trên trình duyệt này.");
      return;
    }

    try {
      previewObjectUrl = URL.createObjectURL(file);
      previewVideo.hidden = false;
      previewVideo.controls = false;
      previewVideo.classList.add("video-decoder-hidden");

      const metadataReady = waitForVideoEvent("loadedmetadata");
      previewVideo.src = previewObjectUrl;
      previewVideo.load();
      await metadataReady;
      if (requestId !== validationRequest) return;

      if (!previewVideo.videoWidth || !previewVideo.videoHeight) {
        throw new Error("Video chưa có khung hình hợp lệ.");
      }

      infoFilename.textContent = file.name;
      infoFilename.title = file.name;
      infoResolution.textContent = `${previewVideo.videoWidth} × ${previewVideo.videoHeight}`;
      infoDuration.textContent = formatDuration(previewVideo.duration);
      infoFormat.textContent = extension.toUpperCase();

      if (previewVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        const firstFrameReady = waitForVideoEvent("loadeddata");
        previewVideo.currentTime = 0;
        await firstFrameReady;
      }
      if (requestId !== validationRequest) return;

      drawVideoFrame(false);
      previewCanvas.hidden = false;
      previewEmpty.hidden = true;
      activeMediaKind = "video";
      setAppState("Đã nạp");
      fileFeedback.textContent = `Đã tải video hợp lệ: ${file.name}`;
      fileFeedback.classList.add("is-success");
      fileFeedback.hidden = false;
    } catch {
      if (requestId === validationRequest) {
        showFileError("Định dạng video không được hỗ trợ hoặc tệp video bị lỗi.");
      }
    }
  }

  function waitForVideoEvent(eventName) {
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => finish(reject, new Error("Đã hết thời gian chờ video.")), 30000);
      const onSuccess = () => finish(resolve);
      const onError = () => finish(reject, new Error("Không thể đọc video."));

      function finish(callback, value) {
        window.clearTimeout(timeout);
        previewVideo.removeEventListener(eventName, onSuccess);
        previewVideo.removeEventListener("error", onError);
        callback(value);
      }

      previewVideo.addEventListener(eventName, onSuccess, { once: true });
      previewVideo.addEventListener("error", onError, { once: true });
    });
  }

  function formatDuration(duration) {
    if (!Number.isFinite(duration) || duration < 0) return "Không xác định";
    const totalSeconds = Math.floor(duration);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return hours > 0
      ? [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":")
      : [minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
  }

  function clearPreview() {
    stopCanvasRecording(true);
    if (progressAnimationFrame !== null) cancelAnimationFrame(progressAnimationFrame);
    progressAnimationFrame = null;
    stopVideoFrameLoop();
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    previewVideo.pause();
    previewVideo.srcObject = null;
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.hidden = true;
    previewVideo.controls = true;
    previewVideo.muted = false;
    previewVideo.autoplay = false;
    previewVideo.classList.remove("video-decoder-hidden");
    previewImage.removeAttribute("src");
    previewImage.hidden = true;
    previewCanvas.hidden = true;
    previewEmpty.hidden = false;
    activeMediaKind = null;
    previewEmptyTitle.textContent = "Chưa có tệp được tải lên";
    previewEmptyDescription.textContent = "Chọn tệp để xem trước hình ảnh hoặc video tại đây.";
    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
    mediaStatus.lastChild.textContent = "Chưa có phương tiện";
    infoFilename.textContent = "Chưa có tệp";
    infoFilename.removeAttribute("title");
    infoResolution.textContent = "—";
    infoDuration.textContent = "—";
    infoFormat.textContent = "—";
    setProgress(0);
  }

  function showFileError(message) {
    stopCanvasRecording(true);
    if (progressAnimationFrame !== null) cancelAnimationFrame(progressAnimationFrame);
    progressAnimationFrame = null;
    stopVideoFrameLoop();
    setProgress(0);
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
    previewVideo.pause();
    previewVideo.srcObject = null;
    previewVideo.removeAttribute("src");
    previewVideo.load();
    previewVideo.hidden = true;
    previewVideo.classList.remove("video-decoder-hidden");
    previewVideo.classList.remove("video-decoder-hidden");
    previewImage.removeAttribute("src");
    previewImage.hidden = true;
    previewCanvas.hidden = true;
    previewEmpty.hidden = false;
    activeMediaKind = null;
    previewEmptyTitle.textContent = "Lỗi camera hoặc tệp";
    previewEmptyDescription.textContent = message;
    setAppState("Lỗi");
    fileFeedback.textContent = message;
    fileFeedback.classList.remove("is-success");
    fileFeedback.hidden = false;
    infoFilename.textContent = "Chưa có tệp";
    infoFilename.removeAttribute("title");
    infoResolution.textContent = "—";
    infoDuration.textContent = "—";
    infoFormat.textContent = "—";
  }

  async function startWebcam() {
    validationRequest += 1;
    const requestId = validationRequest;
    fileFeedback.hidden = true;
    fileFeedback.classList.remove("is-success");
    clearPreview();
    setAppState("Sẵn sàng");
    previewEmptyTitle.textContent = "Đang tải webcam...";
    previewEmptyDescription.textContent = "Vui lòng cho phép trình duyệt truy cập camera.";
    mediaStatus.lastChild.textContent = "Đang tải webcam...";

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        const unsupported = new Error("Trình duyệt không hỗ trợ truy cập webcam hoặc trang chưa chạy trên HTTPS/localhost.");
        unsupported.name = "NotSupportedError";
        throw unsupported;
      }

      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (requestId !== validationRequest) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      webcamStream = stream;
      previewVideo.srcObject = stream;
      previewVideo.controls = false;
      previewVideo.muted = true;
      previewVideo.autoplay = true;
      previewVideo.playsInline = true;
      previewVideo.hidden = false;
      await previewVideo.play();
      if (requestId !== validationRequest) return;

      previewEmpty.hidden = true;
      activeMediaKind = "webcam";
      setAppState("Đã nạp");
      fileFeedback.textContent = "Webcam đã kết nối thành công.";
      fileFeedback.classList.add("is-success");
      fileFeedback.hidden = false;
      infoFilename.textContent = "Webcam trực tiếp";
      infoFilename.title = "Webcam trực tiếp";
      infoResolution.textContent = `${previewVideo.videoWidth} × ${previewVideo.videoHeight}`;
      infoDuration.textContent = "Trực tiếp";
      infoFormat.textContent = "WEBRTC";
    } catch (error) {
      if (requestId !== validationRequest) return;
      const explanation = getCameraErrorMessage(error);
      if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        webcamStream = null;
      }
      previewVideo.pause();
      previewVideo.srcObject = null;
      previewVideo.hidden = true;
      previewEmpty.hidden = false;
      activeMediaKind = null;
      previewEmptyTitle.textContent = "Lỗi camera";
      previewEmptyDescription.textContent = explanation;
      setAppState("Lỗi");
      mediaStatus.lastChild.textContent = "Lỗi camera";
      fileFeedback.textContent = `Lỗi camera: ${explanation}`;
      fileFeedback.classList.remove("is-success");
      fileFeedback.hidden = false;
      infoFilename.textContent = "Chưa có tệp";
      infoFilename.removeAttribute("title");
      infoResolution.textContent = "—";
      infoDuration.textContent = "—";
      infoFormat.textContent = "—";
    }
  }

  function getCameraErrorMessage(error) {
    switch (error?.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        return "Quyền truy cập camera đã bị từ chối. Hãy cấp quyền camera trong trình duyệt rồi thử lại.";
      case "NotFoundError":
      case "DevicesNotFoundError":
        return "Không tìm thấy webcam được kết nối với thiết bị.";
      case "NotReadableError":
      case "TrackStartError":
        return "Camera đang được ứng dụng khác sử dụng hoặc không thể khởi động.";
      case "OverconstrainedError":
        return "Webcam không hỗ trợ cấu hình video được yêu cầu.";
      case "NotSupportedError":
        return error.message;
      case "SecurityError":
        return "Trình duyệt chặn camera. Hãy mở ứng dụng trên HTTPS hoặc localhost.";
      default:
        return "Không thể mở webcam. Hãy kiểm tra kết nối camera và thử lại.";
    }
  }

  function stopWebcam() {
    if (!webcamStream) return;
    validationRequest += 1;
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
    previewVideo.pause();
    previewVideo.srcObject = null;
    previewVideo.hidden = true;
    previewEmpty.hidden = false;
    activeMediaKind = null;
    previewEmptyTitle.textContent = "Webcam đã dừng";
    previewEmptyDescription.textContent = "Chọn Webcam để kết nối camera lại.";
    fileFeedback.hidden = true;
    infoFilename.textContent = "Chưa có tệp";
    infoFilename.removeAttribute("title");
    infoResolution.textContent = "—";
    infoDuration.textContent = "—";
    infoFormat.textContent = "—";
    setProgress(0);
    setAppState("Đã dừng");
  }

  function setAppState(nextState) {
    appState = nextState;
    const stateClasses = ["is-error", "is-ready", "is-loaded", "is-processing", "is-complete", "is-stopped"];
    mediaStatus.classList.remove(...stateClasses);
    appStateLabel.classList.remove(...stateClasses);
    const stateClass = {
      "Lỗi": "is-error",
      "Đã nạp": "is-loaded",
      "Đang xử lý": "is-processing",
      "Hoàn tất": "is-complete",
      "Đã dừng": "is-stopped",
    }[nextState];
    if (stateClass) {
      mediaStatus.classList.add(stateClass);
      appStateLabel.classList.add(stateClass);
    }
    mediaStatus.lastChild.textContent = nextState;
    appStateLabel.textContent = nextState;

    const canRun = Boolean(activeMediaKind) && ["Đã nạp", "Đã dừng"].includes(nextState);
    runVideoButton.disabled = !canRun;
    stopVideoButton.disabled = nextState !== "Đang xử lý" && !webcamStream;
    exportVideoButton.disabled = true;
    exportVideoButton.classList.toggle("is-ready", nextState === "Hoàn tất");
  }

  function setProgress(value) {
    const progress = Math.max(0, Math.min(100, Math.round(value)));
    processingProgress.value = progress;
    progressValue.textContent = `${progress}%`;
  }

  function startProcessing() {
    if (!activeMediaKind || !["Đã nạp", "Đã dừng"].includes(appState)) return;
    if (activeMediaKind === "video") {
      void startVideoProcessing();
      return;
    }
    const initialProgress = appState === "Đã dừng" ? Number(processingProgress.value) : 0;
    const startTime = performance.now();
    setProgress(initialProgress);
    setAppState("Đang xử lý");

    const advance = (now) => {
      const elapsed = now - startTime;
      const nextProgress = initialProgress + (elapsed / 4000) * (100 - initialProgress);
      setProgress(nextProgress);
      if (nextProgress >= 100) {
        progressAnimationFrame = null;
        setProgress(100);
        runSimulatedDetection();
        setAppState("Hoàn tất");
        return;
      }
      progressAnimationFrame = requestAnimationFrame(advance);
    };

    progressAnimationFrame = requestAnimationFrame(advance);
  }

  function stopProcessingOrWebcam() {
    if (appState === "Đang xử lý") {
      if (progressAnimationFrame !== null) cancelAnimationFrame(progressAnimationFrame);
      progressAnimationFrame = null;
      if (activeMediaKind === "video") {
        previewVideo.pause();
        stopVideoFrameLoop();
      }
      setAppState("Đã dừng");
      return;
    }
    if (webcamStream) stopWebcam();
  }

  async function startVideoProcessing() {
    if (previewVideo.ended || (Number.isFinite(previewVideo.duration) && previewVideo.currentTime >= previewVideo.duration)) {
      previewVideo.currentTime = 0;
    }
    setAppState("Đang xử lý");
    renderSimulatedResults(simulatedDetectionData);

    try {
      try {
        startOrResumeCanvasRecording();
      } catch (recordingError) {
        stopCanvasRecording(true);
        fileFeedback.textContent = `Video vẫn có thể phát, nhưng không thể ghi để xuất: ${recordingError.message}`;
        fileFeedback.classList.remove('is-success');
        fileFeedback.hidden = false;
      }
      await previewVideo.play();
      if (appState !== "Đang xử lý") {
        previewVideo.pause();
        return;
      }
      startVideoFrameLoop();
      updateVideoProgress();
    } catch {
      if (appState === "Đang xử lý") {
        showFileError("Không thể phát video trong trình duyệt này.");
      }
    }
  }

  function updateVideoProgress() {
    if (appState !== "Đang xử lý") return;
    const duration = previewVideo.duration;
    if (Number.isFinite(duration) && duration > 0) {
      setProgress((previewVideo.currentTime / duration) * 100);
    }
    if (previewVideo.ended) {
      finishVideoProcessing();
    }
  }

  function finishVideoProcessing() {
    if (appState === "Hoàn tất") return;
    if (activeMediaKind !== "video") return;
    if (progressAnimationFrame !== null) cancelAnimationFrame(progressAnimationFrame);
    progressAnimationFrame = null;
    stopVideoFrameLoop();
    drawVideoFrame(true); stopCanvasRecording();
    setProgress(100);
    setAppState("Hoàn tất");
  }

  function startVideoFrameLoop() {
    stopVideoFrameLoop();
    const loopToken = videoFrameLoopToken;
    let frames = 0;
    let sampleStart = performance.now();
    const renderFrame = () => {
      if (loopToken !== videoFrameLoopToken || activeMediaKind !== "video") return;
      drawVideoFrame(appState === "Đang xử lý" || appState === "Hoàn tất");
      frames += 1;
      const elapsed = performance.now() - sampleStart;
      if (elapsed >= 1000) {
        document.querySelector('#render-fps').textContent = Math.round(frames * 1000 / elapsed);
        frames = 0;
        sampleStart = performance.now();
      }
      if (previewVideo.paused || previewVideo.ended) return;

      if (typeof previewVideo.requestVideoFrameCallback === "function") {
        videoFrameCallbackId = previewVideo.requestVideoFrameCallback(renderFrame);
      } else {
        videoFrameAnimationFrame = requestAnimationFrame(renderFrame);
      }
    };
    renderFrame();
  }

  function stopVideoFrameLoop() {
    videoFrameLoopToken += 1;
    if (videoFrameCallbackId !== null && typeof previewVideo.cancelVideoFrameCallback === "function") {
      previewVideo.cancelVideoFrameCallback(videoFrameCallbackId);
    }
    if (videoFrameAnimationFrame !== null) cancelAnimationFrame(videoFrameAnimationFrame);
    videoFrameCallbackId = null;
    videoFrameAnimationFrame = null;
  }

  function drawVideoFrame(includeDetections) {
    const width = previewVideo.videoWidth;
    const height = previewVideo.videoHeight;
    if (!width || !height || previewVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    if (previewCanvas.width !== width || previewCanvas.height !== height) {
      previewCanvas.width = width;
      previewCanvas.height = height;
    }
    const context = previewCanvas.getContext("2d");
    if (!context) return;
    try {
      context.drawImage(previewVideo, 0, 0, width, height);
    } catch {
      return;
    }
    if (includeDetections) drawBoundingBoxes(context, width, height);
  }

  function drawBoundingBoxes(context, mediaWidth, mediaHeight) {
    if (typeof uiSettings !== "undefined" && !uiSettings.boxes) return;
    const scaleX = mediaWidth / 640;
    const scaleY = mediaHeight / 480;
    const lineScale = Math.min(scaleX, scaleY);
    context.lineWidth = Math.max(3, 3 * lineScale);
    context.font = `600 ${Math.max(16, Math.round(16 * lineScale))}px "Be Vietnam Pro", sans-serif`;
    context.textBaseline = "middle";

    for (const detection of simulatedDetectionData.detections) {
      const [x, y, width, height] = detection.bbox;
      const boxX = x * scaleX;
      const boxY = y * scaleY;
      const boxWidth = width * scaleX;
      const boxHeight = height * scaleY;
      const color = detection.class_name === "Nhựa" ? "#20c997" : "#f4a340";
      const label = `#${detection.track_id} ${detection.class_name}${typeof uiSettings === "undefined" || uiSettings.confidence ? " " + detection.confidence.toFixed(2) : ""}`;
      const fontSize = Math.max(16, Math.round(16 * lineScale));
      const labelHeight = fontSize * 1.6;
      const labelWidth = context.measureText(label).width + fontSize;
      const labelY = Math.max(0, boxY - labelHeight);
      context.strokeStyle = color;
      context.fillStyle = color;
      context.strokeRect(boxX, boxY, boxWidth, boxHeight);
      context.fillRect(boxX, labelY, labelWidth, labelHeight);
      context.fillStyle = "#ffffff";
      context.fillText(label, boxX + fontSize / 2, labelY + labelHeight / 2);
    }
  }

  function runSimulatedDetection() {
    if (!activeMediaKind) return;
    const media = activeMediaKind === "image" ? previewImage : previewVideo;
    const mediaWidth = activeMediaKind === "image" ? media.naturalWidth : media.videoWidth;
    const mediaHeight = activeMediaKind === "image" ? media.naturalHeight : media.videoHeight;
    if (!mediaWidth || !mediaHeight) return;

    const canvas = previewCanvas;
    const context = canvas.getContext("2d");
    if (!context) return;
    canvas.width = mediaWidth;
    canvas.height = mediaHeight;
    context.clearRect(0, 0, canvas.width, canvas.height);
    drawBoundingBoxes(context, mediaWidth, mediaHeight);

    canvas.hidden = false;
    renderSimulatedResults(simulatedDetectionData);
  }

  function renderSimulatedResults(data) {
    resultsList.replaceChildren();

    for (const detection of data.detections) {
      const item = document.createElement("article");
      item.className = "result-item";

      const heading = document.createElement("div");
      heading.className = "result-item-heading";
      const category = document.createElement("span");
      category.className = "category-name";
      const dot = document.createElement("span");
      dot.className = `category-dot ${detection.class_name === "Nhựa" ? "plastic" : "paper"}`;
      dot.setAttribute("aria-hidden", "true");
      category.append(dot, document.createTextNode(detection.class_name));

      const score = document.createElement("strong");
      score.textContent = `${Math.round(detection.confidence * 100)}%`;
      heading.append(category, score);

      const track = document.createElement("div");
      track.className = "confidence-track";
      track.setAttribute("role", "progressbar");
      track.setAttribute("aria-label", `Độ tin cậy ${detection.class_name}`);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", String(Math.round(detection.confidence * 100)));
      const fill = document.createElement("span");
      fill.className = `confidence-fill ${detection.class_name === "Nhựa" ? "plastic-fill" : "paper-fill"}`;
      fill.style.width = `${detection.confidence * 100}%`;
      track.append(fill);

      const trackId = document.createElement("small");
      trackId.className = "detection-track-id";
      trackId.textContent = `Mã theo dõi #${detection.track_id}`;
      item.append(heading, track, trackId);
      resultsList.append(item);
    }

    resultsNote.textContent = `Khung hình #${data.frame_id} · Kết quả giả lập, chưa phải phân tích thực tế.`;
  }
}
// Record the rendered canvas so exported video includes both the source frames and detections.
let canvasRecorder = null;
let canvasRecordingStream = null;
let canvasRecordingBlob = null;
let canvasRecordingUrl = null;

function startOrResumeCanvasRecording() {
  if (canvasRecorder?.state === "paused") {
    canvasRecorder.resume();
    return;
  }
  if (canvasRecorder?.state === "recording") return;

  const canvas = document.querySelector("canvas.media-layer") || document.querySelector("canvas");
  if (!canvas?.captureStream || typeof MediaRecorder === "undefined") {
    throw new Error("Trình duyệt không hỗ trợ ghi video từ Canvas.");
  }
  if (typeof MediaRecorder.isTypeSupported === "function" && !MediaRecorder.isTypeSupported("video/webm")) {
    throw new Error("Trình duyệt không hỗ trợ định dạng video/webm.");
  }

  recordedChunks.length = 0;
  canvasRecordingBlob = null;
  if (canvasRecordingUrl) URL.revokeObjectURL(canvasRecordingUrl);
  canvasRecordingUrl = null;
  canvasRecordingStream = canvas.captureStream(30);
  canvasRecorder = new MediaRecorder(canvasRecordingStream, { mimeType: "video/webm" });
  const activeRecorder = canvasRecorder;
  activeRecorder.ondataavailable = (event) => {
    if (activeRecorder === canvasRecorder && event.data && event.data.size > 0) recordedChunks.push(event.data);
  };
  activeRecorder.onstop = () => {
    if (activeRecorder !== canvasRecorder) return;
    canvasRecordingBlob = new Blob(recordedChunks, { type: "video/webm" });
    if (canvasRecordingBlob.size > 0) {
      canvasRecordingUrl = URL.createObjectURL(canvasRecordingBlob);
      const exportButton = document.getElementById("export-video");
      if (exportButton) exportButton.disabled = false;
    }
    canvasRecordingStream?.getTracks().forEach((track) => track.stop());
    canvasRecordingStream = null;
  };
  activeRecorder.start(1000);
}

function pauseCanvasRecording() {
  if (canvasRecorder?.state === "recording") canvasRecorder.pause();
}

function stopCanvasRecording(discard = false) {
  const recorder = canvasRecorder;
  if (recorder && recorder.state !== "inactive") recorder.stop();
  if (discard) {
    canvasRecorder = null;
    canvasRecordingStream?.getTracks().forEach((track) => track.stop());
    canvasRecordingStream = null;
    recordedChunks.length = 0;
    canvasRecordingBlob = null;
    if (canvasRecordingUrl) URL.revokeObjectURL(canvasRecordingUrl);
    canvasRecordingUrl = null;
  }
}

function downloadRecordedVideo() {
  if (!canvasRecordingBlob || canvasRecorder?.state !== "inactive" || !recordedChunks.length) return;
  const blob = new Blob(recordedChunks, { type: "video/webm" });
  if (!blob.size) return;
  const downloadUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = "ket_qua_thu_nghiem.webm";
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
}

document.getElementById("stop-video")?.addEventListener("click", pauseCanvasRecording);
window.addEventListener('beforeunload', () => stopCanvasRecording(true));
