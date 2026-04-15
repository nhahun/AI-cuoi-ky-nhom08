const form = document.getElementById("predict-form");
const imageInput = document.getElementById("image-input");
const previewImage = document.getElementById("preview-image");
const previewEmpty = document.getElementById("preview-empty");
const confSlider = document.getElementById("conf");
const confValue = document.getElementById("conf-value");
const statusBox = document.getElementById("status-box");
const resultOriginal = document.getElementById("result-original");
const resultAnnotated = document.getElementById("result-annotated");
const originalEmpty = document.getElementById("original-empty");
const annotatedEmpty = document.getElementById("annotated-empty");
const totalDetections = document.getElementById("total-detections");
const classesFound = document.getElementById("classes-found");
const countsTable = document.getElementById("counts-table");
const detectionsTable = document.getElementById("detections-table");
const modelUsed = document.getElementById("model-used");
const downloadLink = document.getElementById("download-link");
const resetButton = document.getElementById("reset-button");
const cameraStream = document.getElementById("camera-stream");
const cameraCanvas = document.getElementById("camera-canvas");
const startCameraButton = document.getElementById("start-camera");
const stopCameraButton = document.getElementById("stop-camera");
const captureCameraButton = document.getElementById("capture-camera");
const themeToggleButton = document.getElementById("theme-toggle");

let activeFile = null;
let mediaStream = null;
const THEME_KEY = "waste_demo_theme";

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

async function updatePreview(file) {
  if (!file) {
    previewImage.style.display = "none";
    previewEmpty.style.display = "block";
    previewImage.src = "";
    return;
  }

  const imageUrl = await readFileAsDataURL(file);
  previewImage.src = imageUrl;
  previewImage.style.display = "block";
  previewEmpty.style.display = "none";
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

function renderDetections(detections) {
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
      `
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

function updateResults(payload) {
  resultOriginal.src = `data:image/jpeg;base64,${payload.original_image}`;
  resultAnnotated.src = `data:image/jpeg;base64,${payload.annotated_image}`;
  resultOriginal.style.display = "block";
  resultAnnotated.style.display = "block";
  originalEmpty.style.display = "none";
  annotatedEmpty.style.display = "none";

  totalDetections.textContent = payload.summary.total_detections;
  classesFound.textContent = payload.summary.classes_found;
  modelUsed.textContent = `Model đang sử dụng: ${payload.model_label}`;
  downloadLink.href = resultAnnotated.src;

  renderCounts(payload.counts);
  renderDetections(payload.detections);
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
  await updatePreview(activeFile);
  setStatus("Đã chụp ảnh từ camera. Sẵn sàng dự đoán.", "success");
}

confSlider.addEventListener("input", () => {
  confValue.textContent = confSlider.value;
});

imageInput.addEventListener("change", async (event) => {
  activeFile = event.target.files[0] || null;
  await updatePreview(activeFile);
  if (activeFile) {
    setStatus("Đã chọn ảnh. Bấm 'Dự đoán ngay' để bắt đầu.", "idle");
  }
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
  confValue.textContent = confSlider.value;
  await updatePreview(null);
  resultOriginal.style.display = "none";
  resultAnnotated.style.display = "none";
  resultOriginal.src = "";
  resultAnnotated.src = "";
  originalEmpty.style.display = "grid";
  annotatedEmpty.style.display = "grid";
  totalDetections.textContent = "0";
  classesFound.textContent = "0";
  countsTable.innerHTML = '<div class="table-placeholder">Chưa có dữ liệu.</div>';
  detectionsTable.innerHTML = '<div class="table-placeholder">Chưa có dữ liệu.</div>';
  modelUsed.textContent = "Model đang sử dụng: chưa có";
  downloadLink.removeAttribute("href");
  stopCamera(false);
  setStatus("Đã đặt lại giao diện.", "idle");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!activeFile) {
    setStatus("Vui lòng tải ảnh lên hoặc chụp ảnh trước khi dự đoán.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", activeFile);
  formData.append("conf", document.getElementById("conf").value);
  formData.append("max_det", document.getElementById("max_det").value);
  formData.append("model_id", document.getElementById("model_id").value);

  try {
    setStatus("Đang dự đoán, vui lòng chờ trong giây lát...", "loading");
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Không thể dự đoán.");
    }

    updateResults(payload);
    setStatus("Dự đoán thành công.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

initTheme();
