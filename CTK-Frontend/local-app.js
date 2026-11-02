/* Local API integration. Original app.js/interface.js remain available in Git for comparison. */
(() => {
  'use strict';
  const $ = selector => document.querySelector(selector);
  const el = (tag, text, className) => Object.assign(document.createElement(tag), {textContent: text || '', className: className || ''});
  const choose = $('#choose-file'), run = $('#run-video'), stop = $('#stop-video'), exportButton = $('#export-video');
  const input = $('#image-file-input'), image = $('#preview-image'), video = $('#preview-video'), canvas = $('canvas.media-layer');
  const empty = $('#preview-empty'), feedback = $('#file-feedback'), resultList = $('#results-list');
  const saveImage = $('#save-image'), dialog = $('#utility-dialog'), dialogBody = $('#dialog-body');
  let selected = null, objectURL = null, currentJob = null, camera = null, busy = false, view = 'home', pollToken = 0, server = null;
  const configuration = el('div', '', 'local-config');
  configuration.innerHTML = `<label>Mô hình<select id="model-select" aria-label="Mô hình"></select></label><label>Chế độ<select id="mode-select"><option value="track">YOLO + ByteTrack (video)</option><option value="predict">YOLO</option></select></label><label>Thiết bị<select id="device-select"><option value="cpu">CPU</option><option value="0">GPU CUDA</option></select></label><label>Kích thước<select id="size-select"><option>320</option><option>480</option><option>640</option></select></label><label>Ngưỡng tin cậy<input id="conf-select" type="number" min="0.01" max="0.99" step="0.05" value="0.25"></label><button class="toolbar-button" id="reload-models">Tải lại mô hình</button>`;
  $('.source-tabs').before(configuration);
  const notice = el('p', 'Đang kiểm tra máy chủ và trọng số…', 'local-notice');
  notice.id = 'backend-status';
  configuration.before(notice);
  const samples = el('div', '', 'sample-row');
  configuration.after(samples);
  const reports = el('div', '', 'report-actions');
  for (const [label, url] of [['Bảng tốc độ CSV', '/api/benchmark.csv'], ['Biểu đồ tốc độ', '/api/benchmark.png'], ['Tải báo cáo tuần 8', '/api/report.md']]) {
    const a = el('a', label, 'toolbar-button'); a.href = url; a.target = '_blank'; a.rel = 'noopener'; reports.append(a);
  }
  $('.studio-guide').before(reports);
  const metrics = el('div', '', 'run-metrics');
  metrics.innerHTML = '<div>FPS pipeline<strong id="pipeline-fps">—</strong></div><div>Thời gian xử lý<strong id="processing-time">—</strong></div><div>Ghi / hoàn tất tệp<strong id="encoding-time">—</strong></div><div>ID track duy nhất<strong id="track-count">—</strong></div>';
  $('.result-summary').after(metrics);
  const links = el('div', '', 'run-links');
  $('.result-note').before(links);
  const cancel = el('button', 'Hủy tác vụ', 'toolbar-button'); cancel.id = 'cancel-job'; cancel.disabled = true;
  stop.after(cancel);
  $('#result-title').textContent = 'Kết quả mô hình';
  $('.result-summary div:last-child span').textContent = 'FPS suy luận + tracker';
  $('.result-intro').textContent = 'Dự đoán ở khung hình gần nhất';
  $('#results-note').textContent = 'Chạy mô hình để xem dự đoán thật. Độ tin cậy không phải Precision.';
  $('.workspace-intro .mode-badge').textContent = 'API trên laptop';
  $('.source-tabs > span').textContent = 'Xử lý cục bộ · không gửi ảnh lên dịch vụ AI';
  $('.header-context > span:nth-child(2)').textContent = 'YOLO · ByteTrack · Local';
  $('.project-status > div p').textContent = 'Luồng hiện tại: ảnh/video → YOLO segmentation → ByteTrack (video) → hiển thị và xuất kết quả.';
  $('.project-status .mode-badge').textContent = 'Kết nối mô hình thật';
  $('.project-status > p').textContent = 'Trọng số COCO chỉ dùng thử kỹ thuật. Để báo cáo nhận diện rác, dùng best.pt đã huấn luyện và đánh giá trên tập test đã khóa. OpenCLIP chưa tích hợp.';
  $('.home-features article:last-child p').textContent = 'Mở webcam, chụp một khung hình và gửi đến mô hình cục bộ để nhận diện.';
  $('.home-features article:nth-child(2) p').textContent = 'Xử lý toàn bộ video, tạm dừng/tiếp tục tác vụ và tải MP4 có chú thích.';

  async function api(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${response.status}`);
    }
    return response.json();
  }
  function message(text, error = false) {
    feedback.textContent = text; feedback.hidden = false; feedback.classList.toggle('is-success', !error);
  }
  function state(label) {
    $('#app-state-label').textContent = label;
    $('.media-status').lastChild.textContent = label;
  }
  function progress(value) {
    $('#processing-progress').value = value; $('#progress-value').textContent = `${value}%`;
  }
  function controls() {
    run.disabled = !server?.ready || (!selected && !camera) || (busy && currentJob?.status !== 'paused');
    run.textContent = currentJob?.status === 'paused' ? '▶ Tiếp tục' : camera ? '◉ Chụp và nhận diện' : '▶ Chạy';
    stop.disabled = !busy && !camera;
    stop.textContent = busy && currentJob?.kind === 'video' ? 'Ⅱ Tạm dừng' : '■ Dừng';
    cancel.disabled = !busy;
    choose.disabled = busy; $('#start-webcam').disabled = busy;
    configuration.querySelectorAll('input,select,button').forEach(e => e.disabled = busy);
    samples.querySelectorAll('button').forEach(e => e.disabled = busy);
    exportButton.disabled = currentJob?.status !== 'completed' || currentJob?.kind !== 'video';
    saveImage.disabled = currentJob?.status !== 'completed' || currentJob?.kind !== 'image';
  }
  async function loadStatus() {
    try {
      server = await api('/api/status');
      const modelSelect = $('#model-select'), old = modelSelect.value;
      modelSelect.replaceChildren();
      for (const path of server.models) {const option = el('option', path); option.value = path; modelSelect.append(option);}
      if (server.models.includes(old)) modelSelect.value = old;
      $('#device-select option[value="0"]').disabled = !server.machine.cuda_available;
      notice.textContent = `${server.machine.gpu || 'CPU'} · Python ${server.machine.python}. ${server.notice}`;
      if (!server.ready) notice.textContent += ' Chưa có trọng số: chép best.pt vào thư mục models/ và nhấn Tải lại mô hình.';
      if (/yolov12n-seg/.test(modelSelect.value)) notice.textContent += ' Đang chọn trọng số COCO 80 lớp để thử kỹ thuật.';
      controls();
    } catch (error) {
      server = null; notice.textContent = `Không kết nối API. Khởi động bằng start_local.ps1. ${error.message}`; controls();
    }
  }
  $('#reload-models').onclick = loadStatus;
  $('#model-select').onchange = () => {
    notice.classList.remove('model-ok');
    notice.textContent = `Đã chọn ${$('#model-select').value}. Tên lớp thật sẽ được kiểm tra khi chạy; checkpoint cần khớp đủ 16 lớp rác. OpenCLIP chưa tích hợp.`;
  };
  function stopCamera() {
    camera?.getTracks().forEach(track => track.stop()); camera = null; video.srcObject = null;
  }
  function resetMedia() {
    stopCamera(); video.pause(); video.removeAttribute('src'); video.load(); video.hidden = true;
    image.hidden = true; canvas.hidden = true; empty.hidden = false;
    if (objectURL) URL.revokeObjectURL(objectURL); objectURL = null;
    currentJob = null; links.replaceChildren(); resultList.replaceChildren();
    $('#object-count').textContent = '0'; $('#render-fps').textContent = '—';
    for (const id of ['pipeline-fps','processing-time','encoding-time','track-count']) $(`#${id}`).textContent = '—';
    saveImage.disabled = true; exportButton.disabled = true; progress(0);
  }
  async function selectFile(file) {
    if (busy) return;
    const extension = file.name.split('.').pop().toLowerCase();
    const allowed = view === 'video' ? ['mp4','avi','webm'] : ['jpg','jpeg','png'];
    if (!allowed.includes(extension)) {message(view === 'video' ? 'Hãy chọn video MP4, WebM hoặc AVI.' : 'Hãy chọn ảnh JPG/PNG. Với video, mở mục Video.', true); return;}
    resetMedia(); selected = file; objectURL = URL.createObjectURL(file);
    $('#info-filename').textContent = file.name; $('#info-format').textContent = extension.toUpperCase();
    $('#info-resolution').textContent = 'Đang đọc…'; $('#info-duration').textContent = '—';
    if (view === 'video') {
      video.src = objectURL; video.hidden = false; video.controls = true;
      video.onloadedmetadata = () => {$('#info-resolution').textContent = `${video.videoWidth} × ${video.videoHeight}`; $('#info-duration').textContent = `${video.duration.toFixed(2)} s`;};
      video.onerror = () => message('Trình duyệt chưa xem trước được tệp này; có thể thử xử lý bằng backend.', true);
    } else {
      image.src = objectURL; image.hidden = false;
      image.onload = () => {$('#info-resolution').textContent = `${image.naturalWidth} × ${image.naturalHeight}`;};
      image.onerror = () => {selected = null; message('Ảnh hỏng hoặc không đọc được.', true); controls();};
    }
    empty.hidden = true; state('Đã nạp'); message(`Đã nạp ${file.name}. Nhấn Chạy để xử lý bằng mô hình.`); controls();
  }
  choose.onclick = () => input.click();
  input.onchange = () => {const file = input.files[0]; input.value = ''; if (file) void selectFile(file);};
  const drop = $('.preview-frame');
  for (const event of ['dragenter','dragover']) drop.addEventListener(event, e => {e.preventDefault(); drop.classList.add('drag-over');});
  for (const event of ['dragleave','drop']) drop.addEventListener(event, e => {e.preventDefault(); drop.classList.remove('drag-over');});
  drop.addEventListener('drop', e => {if (e.dataTransfer.files[0]) void selectFile(e.dataTransfer.files[0]);});

  $('#start-webcam').onclick = async () => {
    if (busy) return;
    resetMedia(); selected = null;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Webcam cần localhost hoặc HTTPS.');
      camera = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
      video.srcObject = camera; video.hidden = false; video.muted = true; video.controls = false; await video.play();
      empty.hidden = true; $('#info-filename').textContent = 'Webcam · chụp từng khung';
      $('#info-resolution').textContent = `${video.videoWidth} × ${video.videoHeight}`;
      $('#info-duration').textContent = 'Trực tiếp'; $('#info-format').textContent = 'WEBCAM';
      state('Webcam sẵn sàng'); message('Nhấn Chụp và nhận diện để phân tích một khung hình.');
    } catch (error) {stopCamera(); message(`Không mở được camera: ${error.message}`, true); state('Lỗi camera');}
    controls();
  };
  function showJob(job) {
    currentJob = job;
    const labels = {queued:'Đang chờ',running:'Đang xử lý',paused:'Đã tạm dừng',cancelling:'Đang hủy',cancelled:'Đã hủy',completed:'Hoàn tất',failed:'Lỗi',interrupted:'Bị gián đoạn'};
    state(labels[job.status] || job.status); progress(job.progress || 0);
    if (job.preview_url && busy) {
      video.pause(); video.hidden = true; image.hidden = false; empty.hidden = true; image.src = `${job.preview_url}?frame=${job.frames}`;
    }
    if (job.model_note) {notice.textContent = `${job.model} · ${job.classes.length} lớp. ${job.model_note} OpenCLIP chưa tích hợp.`; notice.classList.toggle('model-ok', job.waste_model);}
    $('#object-count').textContent = (job.detections || []).length;
    $('#render-fps').textContent = job.inference_fps?.toFixed(2) || '—';
    $('#pipeline-fps').textContent = job.pipeline_fps?.toFixed(2) || '—';
    $('#processing-time').textContent = job.processing_s != null ? `${job.processing_s.toFixed(3)} s` : '—';
    $('#encoding-time').textContent = job.encoding_s != null ? `${job.encoding_s.toFixed(3)} s` : '—';
    $('#track-count').textContent = job.unique_tracks ?? '—';
    resultList.replaceChildren();
    for (const detection of (job.detections || []).slice(0, 30)) {
      const row = el('article', '', 'result-item'), heading = el('div', '', 'result-item-heading');
      heading.append(el('span', `${detection.track_id == null ? '' : '#'+detection.track_id+' '}${detection.class_name}`), el('strong', `${(detection.confidence*100).toFixed(1)}%`));
      row.append(heading); resultList.append(row);
    }
    if (!(job.detections || []).length) resultList.append(el('p', 'Chưa có vật thể vượt ngưỡng ở khung hiện tại.'));
    $('#results-note').textContent = 'Đây là độ tin cậy dự đoán, không phải Precision. Số ID track chưa phải đếm qua vạch.';
    controls();
  }
  function link(text, href) {const a = el('a', text); a.href = href; a.target = '_blank'; a.rel = 'noopener'; return a;}
  function finish(job) {
    busy = false; showJob(job);
    if (job.status === 'completed') {
      links.replaceChildren(link('CSV dự đoán', job.csv_url), link('Thông tin lần chạy', job.json_url));
      if (job.kind === 'video') {image.hidden = true; video.hidden = false; video.srcObject = null; video.src = job.output_url; video.controls = true; video.muted = false; video.load();}
      else {video.hidden = true; image.hidden = false; image.src = job.output_url;}
      message(`Đã xử lý ${job.frames} khung. Tệp kết quả sẵn sàng. ${job.waste_model ? '' : 'Trọng số chưa khớp 16 lớp rác: chỉ dùng kiểm thử kỹ thuật.'}`);
    } else message(job.error || 'Tác vụ đã hủy; phần video chưa hoàn tất không dùng làm kết quả.', job.status !== 'cancelled');
    controls();
  }
  async function poll(id, token) {
    while (token === pollToken) {
      try {
        const job = await api(`/api/jobs/${id}`);
        if (token !== pollToken) return;
        showJob(job);
        if (['completed','failed','cancelled','interrupted'].includes(job.status)) {finish(job); return;}
      } catch (error) {message(`Mất kết nối khi theo dõi. Tác vụ có thể vẫn chạy; xem Lịch sử. ${error.message}`, true); busy = false; controls(); return;}
      await new Promise(resolve => setTimeout(resolve, 700));
    }
  }
  run.onclick = async () => {
    if (currentJob?.status === 'paused') {try {showJob(await api(`/api/jobs/${currentJob.id}/resume`, {method:'POST'}));} catch(e) {message(e.message,true);} return;}
    if (busy || (!selected && !camera)) return;
    busy = true; currentJob = null; links.replaceChildren(); progress(0); controls(); state('Đang gửi tệp');
    try {
      let file = selected;
      if (camera) {
        const capture = document.createElement('canvas'); capture.width = video.videoWidth; capture.height = video.videoHeight;
        capture.getContext('2d').drawImage(video, 0, 0); const blob = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg'));
        if (!blob) throw new Error('Chưa có khung webcam.');
        file = new File([blob], 'webcam.jpg', {type:'image/jpeg'});
        stopCamera(); selected = file;
      }
      const form = new FormData(); form.append('file', file);
      for (const [field, id] of [['model','model-select'],['mode','mode-select'],['device','device-select'],['imgsz','size-select'],['confidence','conf-select']]) form.append(field, $(`#${id}`).value);
      currentJob = await api('/api/jobs', {method:'POST',body:form});
      message('Mô hình đang xử lý trên máy. Lượt đầu có thêm thời gian nạp và warmup.');
      void poll(currentJob.id, ++pollToken);
    } catch(error) {busy = false; state('Lỗi'); message(error.message, true); controls();}
  };
  stop.onclick = async () => {
    if (busy && currentJob) {
      try {showJob(await api(`/api/jobs/${currentJob.id}/${currentJob.kind === 'video' ? 'pause' : 'cancel'}`, {method:'POST'}));} catch(e) {message(e.message,true);}
    } else {stopCamera(); video.hidden = true; empty.hidden = false; state('Đã dừng webcam'); controls();}
  };
  cancel.onclick = async () => {if (currentJob) try {showJob(await api(`/api/jobs/${currentJob.id}/cancel`, {method:'POST'}));} catch(e) {message(e.message,true);}};
  function downloadResult() {if (currentJob?.output_url) {const a = document.createElement('a'); a.href = currentJob.output_url; a.download = currentJob.kind === 'video' ? `ecovision_${currentJob.id}.mp4` : `ecovision_${currentJob.id}.png`; a.click();}}
  exportButton.onclick = downloadResult; saveImage.onclick = downloadResult;

  function openPanel(title) {$('#dialog-title').textContent = title; dialogBody.replaceChildren(); dialog.showModal();}
  $('#close-dialog').onclick = () => dialog.close();
  async function historyPanel() {
    openPanel('Lịch sử và bằng chứng thực nghiệm');
    try {
      const jobs = await api('/api/jobs');
      if (!jobs.length) dialogBody.append(el('p', 'Chưa có lần chạy. Chọn ảnh test mẫu để bắt đầu.'));
      for (const job of jobs) {
        const row = el('div', '', 'history-row'); row.append(el('strong', `${job.filename} · ${job.status}`));
        row.append(el('p', `${job.created_at} · ${job.model} · ${job.device} · ${job.mode} · ${job.imgsz}px · ID ${job.id}`));
        if (job.output_url) row.append(link('Mở kết quả', job.output_url));
        if (job.json_url) row.append(link('JSON', job.json_url), link('CSV', job.csv_url));
        if (job.error) row.append(el('p', job.error));
        if (['running','queued','paused','cancelling'].includes(job.status)) {
          const follow = el('button','Theo dõi tác vụ','toolbar-button');
          follow.onclick = () => {dialog.close(); busy = true; showJob(job); void poll(job.id, ++pollToken);}; row.append(follow);
        }
        dialogBody.append(row);
      }
    } catch(e) {dialogBody.append(el('p', e.message));}
  }
  function navigate(next) {
    if (!['home','recognition','video'].includes(next)) next = 'home';
    if (busy && next !== view) {message('Hãy hoàn tất hoặc hủy tác vụ trước khi đổi trang.', true); return;}
    if (view !== next) {resetMedia(); selected = null;}
    view = next; $('#home').hidden = view !== 'home'; $('#workspace').hidden = view === 'home'; $('.info-bar').hidden = view === 'home';
    $('.workspace-intro h1').textContent = view === 'video' ? 'Phân tích video' : 'Nhận diện hình ảnh';
    $('.workspace-intro p:last-child').textContent = view === 'video' ? 'Xử lý video bằng YOLO/ByteTrack và xuất MP4 cùng bảng thời gian.' : 'Chọn ảnh test hoặc chụp webcam để chạy mô hình thật.';
    input.accept = view === 'video' ? '.mp4,.avi,.webm' : '.jpg,.jpeg,.png';
    $('[data-source="all"]').textContent = view === 'video' ? 'Chọn video' : 'Chọn hình ảnh';
    $('[data-source="camera"]').hidden = view === 'video';
    samples.dataset.kind = view === 'video' ? 'video' : 'image';
    samples.querySelectorAll('button').forEach(button => button.hidden = button.dataset.kind !== samples.dataset.kind);
    $('#start-webcam').hidden = view === 'video'; exportButton.hidden = view !== 'video'; saveImage.hidden = view === 'video';
    empty.querySelector('strong').textContent = view === 'video' ? 'Thêm video để bắt đầu' : 'Thêm hình ảnh để bắt đầu';
    empty.querySelector('span:last-child').textContent = view === 'video' ? 'Chọn MP4, WebM hoặc AVI; kết quả được xuất thành MP4.' : 'Chọn JPG/PNG hoặc dùng ảnh test mẫu bên trên.';
    $('.preview-footer > span:last-child').textContent = view === 'video' ? 'MP4 · WEBM · AVI' : 'PNG · JPG · JPEG';
    document.querySelectorAll('.primary-nav .nav-link').forEach((a, i) => {const active = ['home','recognition','video'][i] === view; a.classList.toggle('is-active', active); if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
    if (location.hash !== '#'+view) history.replaceState(null,'','#'+view);
    state('Sẵn sàng'); controls();
  }
  document.querySelectorAll('.nav-link').forEach(a => a.onclick = e => {
    e.preventDefault(); const label = a.textContent;
    if (label.includes('Lịch sử')) void historyPanel();
    else if (label.includes('Cài đặt')) {navigate('recognition'); configuration.scrollIntoView({block:'center',behavior:'smooth'});}
    else if (label.includes('Trợ giúp')) {
      openPanel('Hướng dẫn kiểm thử tuần 8');
      dialogBody.append(el('p','1. Chọn mô hình, thiết bị, kích thước và ngưỡng. 2. Nạp ảnh/video và nhấn Chạy. 3. Xem kết quả, tải CSV/JSON và báo cáo. Với video có thể tạm dừng, tiếp tục hoặc hủy tác vụ. MP4 xuất không có âm thanh.'));
      dialogBody.append(el('p','best.pt của nhóm phải được chép vào models/. Mô hình COCO 80 lớp chỉ thử pipeline, không đánh giá nhận diện 16 lớp rác. FPS suy luận + tracker và FPS pipeline có phạm vi đo khác nhau. OpenCLIP chưa tích hợp.'));
    } else navigate(label.includes('Video') ? 'video' : label.includes('Trang chủ') ? 'home' : 'recognition');
  });
  $('.brand').onclick = e => {e.preventDefault(); navigate('home');};
  document.querySelectorAll('.home-actions a').forEach(a => a.onclick = e => {e.preventDefault(); navigate(a.hash.slice(1));});
  document.querySelectorAll('[data-source]').forEach(button => button.onclick = () => button.dataset.source === 'camera' ? $('#start-webcam').click() : choose.click());
  window.addEventListener('hashchange', () => navigate(location.hash.slice(1)));
  window.addEventListener('beforeunload', () => stopCamera());
  function addSamples(items, kind) {items.forEach(item => {
    const button = el('button',item.name,'toolbar-button');
    button.dataset.kind = kind; button.hidden = kind !== (view === 'video' ? 'video' : 'image');
    button.onclick = async () => {try {const response = await fetch(item.url); if(!response.ok)throw new Error('Không tải được tệp mẫu.'); await selectFile(new File([await response.blob()], decodeURIComponent(item.url.split('/').pop()), {type:kind === 'video' ? 'video/mp4' : 'image/jpeg'})); if(kind === 'video')message('Video kỹ thuật tạo từ một ảnh tĩnh, chỉ kiểm tra thao tác/xuất tệp; không dùng đánh giá tracking thực địa.');} catch(e) {message(e.message,true);}};
    samples.append(button);
  });}
  api('/api/samples').then(items => addSamples(items, 'image')).catch(() => {});
  api('/api/fixtures').then(items => addSamples(items, 'video')).catch(() => {});
  navigate(location.hash.slice(1)); void loadStatus();
})();
