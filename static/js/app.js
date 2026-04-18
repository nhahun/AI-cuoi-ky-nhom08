const form = document.getElementById("predict-form");
const mediaInput = document.getElementById("media-input");
const previewImage = document.getElementById("preview-image");
const previewVideo = document.getElementById("preview-video");
const previewEmpty = document.getElementById("preview-empty");
const previewKindBadge = document.getElementById("preview-kind-badge");
const confSlider = document.getElementById("conf");
const confValue = document.getElementById("conf-value");
const statusBox = document.getElementById("status-box");
const resultOriginal = document.getElementById("result-original");
const resultOriginalVideo = document.getElementById("result-original-video");
const resultAnnotated = document.getElementById("result-annotated");
const resultAnnotatedVideo = document.getElementById("result-annotated-video");
const originalEmpty = document.getElementById("original-empty");
const annotatedEmpty = document.getElementById("annotated-empty");
const totalDetections = document.getElementById("total-detections");
const classesFound = document.getElementById("classes-found");
const summaryNote = document.getElementById("summary-note");
const countsTable = document.getElementById("counts-table");
const detectionsTable = document.getElementById("detections-table");
const detailsTitle = document.getElementById("details-title");
const modelUsed = document.getElementById("model-used");
const downloadLink = document.getElementById("download-link");
const resetButton = document.getElementById("reset-button");
const cameraStream = document.getElementById("camera-stream");
const cameraCanvas = document.getElementById("camera-canvas");
const startCameraButton = document.getElementById("start-camera");
const stopCameraButton = document.getElementById("stop-camera");
const captureCameraButton = document.getElementById("capture-camera");
const themeToggleButton = document.getElementById("theme-toggle");
const originalCardTitle = document.getElementById("original-card-title");
const annotatedCardTitle = document.getElementById("annotated-card-title");
const mobileMenuButton = document.getElementById("mobile-menu-toggle");
const mobileNav = document.getElementById("mobile-nav");
const heroVideo = document.getElementById("hero-video");

let activeFile = null;
let activeMediaType = null;
let mediaStream = null;
let previewObjectUrl = null;
let heroRafRef = null;
let heroRestartTimeoutRef = null;

const THEME_KEY = "waste_demo_theme";

function setupHeroVideoLoop() {
  if (!heroVideo) return;

  const fadeWindow = 0.5;

  const safePlay = async () => {
    try {
      await heroVideo.play();
    } catch {
      // Bỏ qua lỗi autoplay.
    }
  };

  const applyOpacity = () => {
    const currentTime = heroVideo.currentTime || 0;
    const duration = heroVideo.duration || 0;
    let opacity = 1;

    if (currentTime < fadeWindow) {
      opacity = currentTime / fadeWindow;
    } else if (duration > fadeWindow && currentTime > duration - fadeWindow) {
      opacity = Math.max((duration - currentTime) / fadeWindow, 0);
    }

    heroVideo.style.opacity = `${Math.min(Math.max(opacity, 0), 1)}`;
  };

  const tick = () => {
    applyOpacity();
    heroRafRef = window.requestAnimationFrame(tick);
  };

  const handleLoadedData = () => {
    heroVideo.style.opacity = "0";
    void safePlay();
  };

  const handleEnded = () => {
    heroVideo.style.opacity = "0";
    if (heroRestartTimeoutRef !== null) {
      window.clearTimeout(heroRestartTimeoutRef);
    }
    heroRestartTimeoutRef = window.setTimeout(() => {
      heroVideo.currentTime = 0;
      void safePlay();
    }, 100);
  };

  heroVideo.loop = false;
  heroVideo.muted = true;
  heroVideo.playsInline = true;
  heroVideo.preload = "auto";
  heroVideo.style.opacity = "0";
  heroVideo.addEventListener("loadeddata", handleLoadedData);
  heroVideo.addEventListener("ended", handleEnded);
  heroRafRef = window.requestAnimationFrame(tick);
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  if (themeToggleButton) {
    themeToggleButton.textContent = theme === "dark" ? "☀️ Chế độ sáng" : "🌙 Chế độ tối";
  }
  localStorage.setItem(THEME_KEY, theme);
}

function initTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY);
  if (savedTheme === "dark" || savedTheme === "light") {
    applyTheme(savedTheme);
    return;
  }
  applyTheme("light");
}

function setStatus(message, type) {
  statusBox.textContent = message;
  statusBox.className = `status-box ${type}`;
}

function detectMediaType(file) {
  if (!file) {
    return null;
  }
  if (file.type.startsWith("video/")) {
    return "video";
  }
  return "image";
}

function fileFromCanvas(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(new File([blob], "camera_capture.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  });
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function clearPreviewObjectUrl() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
}

function hidePreviewMedia() {
  previewImage.style.display = "none";
  previewImage.src = "";
  previewVideo.style.display = "none";
  previewVideo.pause();
  previewVideo.removeAttribute("src");
  previewVideo.load();
}

async function updatePreview(file) {
  clearPreviewObjectUrl();
  hidePreviewMedia();

  if (!file) {
    previewEmpty.style.display = "grid";
    previewKindBadge.textContent = "Preview";
    return;
  }

  activeMediaType = detectMediaType(file);
  previewEmpty.style.display = "none";

  if (activeMediaType === "video") {
    previewObjectUrl = URL.createObjectURL(file);
    previewVideo.src = previewObjectUrl;
    previewVideo.style.display = "block";
    previewKindBadge.textContent = "Video";
    return;
  }

  const imageUrl = await readFileAsDataURL(file);
  previewImage.src = imageUrl;
  previewImage.style.display = "block";
  previewKindBadge.textContent = "Ảnh";
}

function hideResultMedia() {
  [resultOriginal, resultAnnotated].forEach((element) => {
    element.style.display = "none";
    element.src = "";
  });

  [resultOriginalVideo, resultAnnotatedVideo].forEach((element) => {
    element.style.display = "none";
    element.pause();
    element.removeAttribute("src");
    element.load();
  });

  originalEmpty.style.display = "grid";
  annotatedEmpty.style.display = "grid";
  downloadLink.removeAttribute("href");
  downloadLink.textContent = "Tải kết quả";
}

function renderCounts(counts) {
  const entries = Object.entries(counts);
  if (!entries.length) {
    countsTable.innerHTML = '<div class="table-placeholder">Không có vật thể nào được phát hiện.</div>';
    return;
  }

  const rows = entries
    .sort((a, b) => b[1] - a[1])
    .map(([className, count]) => `<tr><td>${className}</td><td>${count}</td></tr>`)
    .join("");

  countsTable.innerHTML = `
    <table class="results-table">
      <thead>
        <tr>
          <th>Class</th>
          <th>Số lượng</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderImageDetections(detections) {
  detailsTitle.textContent = "Danh sách dự đoán";

  if (!detections.length) {
    detectionsTable.innerHTML = '<div class="table-placeholder">Không có dự đoán nào.</div>';
    return;
  }

  const rows = detections
    .map(
      (item, index) => `
        <tr>
          <td>${index + 1}</td>
          <td>${item.class_name}</td>
          <td>${item.confidence}</td>
          <td>[${item.bbox.join(", ")}]</td>
        </tr>
      `,
    )
    .join("");

  detectionsTable.innerHTML = `
    <table class="results-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Class</th>
          <th>Confidence</th>
          <th>Bounding Box</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderVideoDetails(videoSummary) {
  detailsTitle.textContent = "Thống kê toàn video";

  const detailRows = [
    ["Tên file", videoSummary.source_filename],
    ["Số frame đã quét", videoSummary.processed_frames],
    ["Frame ước lượng", videoSummary.estimated_frame_count],
    ["FPS", videoSummary.fps],
    ["Thời lượng (giây)", videoSummary.duration_seconds],
    ["Tổng box trên các frame", videoSummary.frame_level_detections],
    ["Tỉ lệ frame có tracking", `${videoSummary.tracking_coverage_percent}%`],
    ["Chế độ đếm", videoSummary.counting_mode_label],
  ];

  const rows = detailRows.map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join("");

  detectionsTable.innerHTML = `
    <table class="results-table">
      <thead>
        <tr>
          <th>Chỉ số</th>
          <th>Giá trị</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function updateImageResults(payload) {
  originalCardTitle.textContent = "Ảnh đầu vào";
  annotatedCardTitle.textContent = "Ảnh đã nhận diện";

  resultOriginal.src = `data:image/jpeg;base64,${payload.original_image}`;
  resultAnnotated.src = `data:image/jpeg;base64,${payload.annotated_image}`;
  resultOriginal.style.display = "block";
  resultAnnotated.style.display = "block";
  originalEmpty.style.display = "none";
  annotatedEmpty.style.display = "none";

  downloadLink.href = resultAnnotated.src;
  downloadLink.textContent = "Tải ảnh kết quả";
  downloadLink.setAttribute("download", "prediction.jpg");

  renderCounts(payload.counts);
  renderImageDetections(payload.detections);
}

function updateVideoResults(payload) {
  originalCardTitle.textContent = "Video đầu vào";
  annotatedCardTitle.textContent = "Video đã nhận diện";

  resultOriginalVideo.src = payload.original_video_url;
  resultAnnotatedVideo.src = payload.annotated_video_url;
  resultOriginalVideo.style.display = "block";
  resultAnnotatedVideo.style.display = "block";
  originalEmpty.style.display = "none";
  annotatedEmpty.style.display = "none";

  downloadLink.href = payload.annotated_video_url;
  downloadLink.textContent = "Tải video kết quả";
  downloadLink.setAttribute("download", "prediction_video.mp4");

  renderCounts(payload.counts);
  renderVideoDetails(payload.video_summary);
}

function updateResults(payload) {
  hideResultMedia();

  totalDetections.textContent = payload.summary.total_detections;
  classesFound.textContent = payload.summary.classes_found;
  summaryNote.textContent = payload.summary.note;
  modelUsed.textContent = `Model đang sử dụng: ${payload.model_label}`;

  if (payload.media_type === "video") {
    updateVideoResults(payload);
  } else {
    updateImageResults(payload);
  }

  const demoSection = document.getElementById("demo");
  if (demoSection) {
    demoSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Trình duyệt không hỗ trợ camera.", "error");
    return;
  }

  try {
    if (mediaStream) {
      stopCamera(false);
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraStream.srcObject = mediaStream;
    setStatus("Camera đã sẵn sàng. Bạn có thể chụp ảnh.", "success");
  } catch (error) {
    setStatus(`Không thể mở camera: ${error.message}`, "error");
  }
}

function stopCamera(showStatus = true) {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  cameraStream.srcObject = null;
  if (showStatus) {
    setStatus("Camera đã được tắt.", "idle");
  }
}

async function captureFromCamera() {
  if (!mediaStream) {
    setStatus("Vui lòng bật camera trước khi chụp ảnh.", "error");
    return;
  }

  cameraCanvas.width = cameraStream.videoWidth || 1280;
  cameraCanvas.height = cameraStream.videoHeight || 720;
  const context = cameraCanvas.getContext("2d");
  context.drawImage(cameraStream, 0, 0, cameraCanvas.width, cameraCanvas.height);
  activeFile = await fileFromCanvas(cameraCanvas);
  activeMediaType = "image";
  await updatePreview(activeFile);
  setStatus("Đã chụp ảnh từ camera. Sẵn sàng quét.", "success");
}

function setupMobileMenu() {
  if (!mobileMenuButton || !mobileNav) return;

  mobileMenuButton.addEventListener("click", () => {
    const willOpen = !mobileNav.classList.contains("is-open");
    mobileNav.classList.toggle("is-open", willOpen);
    mobileMenuButton.classList.toggle("is-open", willOpen);
  });

  mobileNav.querySelectorAll("a").forEach((anchor) => {
    anchor.addEventListener("click", () => {
      mobileNav.classList.remove("is-open");
      mobileMenuButton.classList.remove("is-open");
    });
  });
}

confSlider.addEventListener("input", () => {
  confValue.textContent = confSlider.value;
});

mediaInput.addEventListener("change", async (event) => {
  activeFile = event.target.files[0] || null;
  activeMediaType = detectMediaType(activeFile);
  await updatePreview(activeFile);

  if (!activeFile) {
    return;
  }

  if (activeMediaType === "video") {
    setStatus("Đã chọn video. Bấm 'Quét ngay' để thống kê toàn bộ clip.", "idle");
    return;
  }

  setStatus("Đã chọn ảnh. Bấm 'Quét ngay' để bắt đầu.", "idle");
});

startCameraButton.addEventListener("click", startCamera);
stopCameraButton.addEventListener("click", () => stopCamera(true));
captureCameraButton.addEventListener("click", captureFromCamera);

if (themeToggleButton) {
  themeToggleButton.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  });
}

resetButton.addEventListener("click", async () => {
  form.reset();
  activeFile = null;
  activeMediaType = null;
  confValue.textContent = confSlider.value;
  await updatePreview(null);
  hideResultMedia();
  totalDetections.textContent = "0";
  classesFound.textContent = "0";
  summaryNote.textContent = "Kết quả dự đoán sẽ cập nhật sau mỗi lần quét.";
  countsTable.innerHTML = '<div class="table-placeholder">Chưa có dữ liệu.</div>';
  detectionsTable.innerHTML = '<div class="table-placeholder">Chưa có dữ liệu.</div>';
  detailsTitle.textContent = "Chi tiết kết quả";
  modelUsed.textContent = "Model đang sử dụng: chưa có";
  downloadLink.removeAttribute("href");
  downloadLink.textContent = "Tải kết quả";
  downloadLink.removeAttribute("download");
  stopCamera(false);
  setStatus("Đã đặt lại giao diện.", "idle");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!activeFile) {
    setStatus("Vui lòng tải ảnh, video hoặc chụp ảnh trước khi quét.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("media", activeFile);
  formData.append("conf", document.getElementById("conf").value);
  formData.append("max_det", document.getElementById("max_det").value);
  formData.append("model_id", document.getElementById("model_id").value);

  try {
    if (activeMediaType === "video") {
      setStatus("Đang quét toàn bộ video, vui lòng chờ trong giây lát...", "loading");
    } else {
      setStatus("Đang quét ảnh, vui lòng chờ trong giây lát...", "loading");
    }

    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Không thể dự đoán.");
    }

    updateResults(payload);
    setStatus("Quét thành công.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

hideResultMedia();
initTheme();
setupHeroVideoLoop();
setupMobileMenu();
