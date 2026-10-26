const uiSettings = { boxes: true, confidence: true };
const sessionHistory = [];
const dialog = document.querySelector('#utility-dialog');
const dialogBody = document.querySelector('#dialog-body');
const saveImageButton = document.querySelector('#save-image');
function openPanel(title, html) {
  document.querySelector('#dialog-title').textContent = title;
  dialogBody.innerHTML = html;
  dialog.showModal();
}
document.querySelector('#close-dialog').onclick = () => dialog.close();
document.querySelectorAll('.nav-link').forEach((link) => link.addEventListener('click', (event) => {
  event.preventDefault();
  const label = link.textContent.trim();
  if (label.includes('Lịch sử')) {
    openPanel('Lịch sử phiên làm việc', '<p>Kết quả giả lập trong phiên hiện tại. Tệp gốc không được lưu.</p>');
    if (!sessionHistory.length) dialogBody.append(Object.assign(document.createElement('p'), {textContent:'Chưa có phiên hoàn tất. Hãy chọn tệp và nhấn Chạy.'}));
    for (const entry of sessionHistory.slice().reverse()) {
      const row = document.createElement('div'); row.className = 'history-row';
      const name = document.createElement('strong'); name.textContent = entry.name;
      const detail = document.createElement('span'); detail.textContent = `${entry.time} · ${entry.count} vật thể · Giả lập`;
      row.append(name, detail); dialogBody.append(row);
    }
  } else if (label.includes('Cài đặt')) {
    openPanel('Cài đặt hiển thị', `<p>Áp dụng khi chạy lần tiếp theo. Bộ suy luận hiện tại sử dụng dữ liệu giả lập.</p><label class="setting-row">Hiển thị khung nhận diện<input id="boxes-setting" type="checkbox" ${uiSettings.boxes?'checked':''}></label><label class="setting-row">Hiển thị điểm tin cậy trên ảnh<input id="confidence-setting" type="checkbox" ${uiSettings.confidence?'checked':''}></label>`);
    document.querySelector('#boxes-setting').onchange = e => uiSettings.boxes = e.target.checked;
    document.querySelector('#confidence-setting').onchange = e => uiSettings.confidence = e.target.checked;
  } else if (label.includes('Trợ giúp')) {
    openPanel('Bắt đầu với EcoVision', '<ol><li>Chọn hoặc kéo thả ảnh PNG/JPG, video MP4/WebM/AVI. Khả năng đọc video phụ thuộc codec trình duyệt.</li><li>Nhấn Chạy để xem kết quả giả lập. Dừng để tạm dừng video, Chạy để tiếp tục.</li><li>Khi hoàn tất, lưu ảnh PNG hoặc xuất video WebM có khung nhận diện (không kèm âm thanh).</li><li>Webcam cần quyền camera và địa chỉ localhost hoặc HTTPS. Chọn tệp mới để kết thúc camera.</li></ol><p>YOLO, ByteTrack và OpenCLIP chưa được kết nối. FPS là tốc độ hiển thị đo được, không phải tốc độ suy luận AI.</p>');
  } else {
    navigateTo(label.includes('Video') ? 'video' : label.includes('Trang chủ') ? 'home' : 'recognition');
  }
}));
document.querySelectorAll('[data-source]').forEach(button => button.onclick = () => {
  document.querySelectorAll('[data-source]').forEach(b => b.classList.toggle('selected',b===button));
  if (button.dataset.source === 'camera') startWebcamButton.click();
  else chooseFileButton.click();
});
const dropZone = document.querySelector('.preview-frame');
for (const name of ['dragenter','dragover']) dropZone.addEventListener(name,e=>{e.preventDefault();dropZone.classList.add('drag-over');});
for (const name of ['dragleave','drop']) dropZone.addEventListener(name,e=>{e.preventDefault();dropZone.classList.remove('drag-over');});
dropZone.addEventListener('drop',e=>{
  if (!e.dataTransfer.files.length) return;
  const transfer = new DataTransfer(); transfer.items.add(e.dataTransfer.files[0]);
  imageFileInput.files = transfer.files; imageFileInput.dispatchEvent(new Event('change'));
});
function syncResultSummary() {
  const complete = appState === 'Hoàn tất';
  document.querySelector('#object-count').textContent = ['Đang xử lý','Hoàn tất','Đã dừng'].includes(appState) && activeMediaKind ? new Set(simulatedDetectionData.detections.map(d=>d.track_id)).size : 0;
  saveImageButton.disabled = !complete || activeMediaKind === 'video';
  if (complete) sessionHistory.push({name: infoFilename.textContent,time:new Date().toLocaleString('vi-VN'),count:simulatedDetectionData.detections.length});
}
new MutationObserver(syncResultSummary).observe(appStateLabel,{childList:true});
saveImageButton.onclick = () => {
  if (saveImageButton.disabled) return;
  const canvas = document.createElement('canvas'); canvas.width = previewCanvas.width; canvas.height = previewCanvas.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(activeMediaKind === 'image' ? previewImage : previewVideo,0,0,canvas.width,canvas.height);
  ctx.drawImage(previewCanvas,0,0);
  canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='ket_qua_thu_nghiem.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);},'image/png');
};
// Keep the initial state empty; demo scores appear only when processing begins.
resultsList.querySelectorAll('.result-item').forEach(item=>{
  item.querySelector('strong').textContent='—';item.querySelector('.confidence-fill').style.width='0%';item.querySelector('.confidence-track').setAttribute('aria-valuenow','0');
});

let currentView = null;
function navigateTo(view) {
  if (!['home','video','recognition'].includes(view)) view = 'home';
  if (currentView !== view) document.dispatchEvent(new Event('workspace:leave'));
  currentView = view;
  document.querySelector('#home').hidden = view !== 'home';
  document.querySelector('#workspace').hidden = view === 'home';
  document.querySelector('.info-bar').hidden = view === 'home';
  document.querySelectorAll('.primary-nav .nav-link').forEach((link,index)=>{
    const active = ['home','recognition','video'][index] === view;
    link.classList.toggle('is-active',active);
    if(active)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');
  });
  const isVideo = view === 'video';
  imageFileInput.accept = isVideo ? '.mp4,.webm,.avi' : '.png,.jpg,.jpeg';
  document.querySelector('#workspace-title').textContent = isVideo ? 'Phân tích video' : 'Nhận diện hình ảnh';
  document.querySelector('.workspace-intro p:last-child').textContent = isVideo ? 'Chọn video, chạy phân tích và xuất kết quả có chú thích.' : 'Tải ảnh hoặc mở webcam để xem kết quả nhận diện.';
  document.querySelector('[data-source="all"]').textContent = isVideo ? 'Chọn video' : 'Chọn hình ảnh';
  document.querySelector('[data-source="camera"]').hidden = isVideo;
  exportVideoButton.hidden = !isVideo;
  saveImageButton.hidden = isVideo;
  document.querySelector('#preview-title').textContent = isVideo ? 'Xem trước video' : 'Xem trước';
  document.querySelector('.preview-footer > span:last-child').textContent = isVideo ? 'MP4 · WEBM · AVI (tùy codec)' : 'PNG · JPG · JPEG';
  document.querySelector('.studio-guide .guide-step p').textContent = isVideo ? 'Chọn tệp hoặc kéo thả video vào khung xem trước.' : 'Chọn ảnh hoặc dùng webcam trực tiếp.';
  const matching = isVideo ? activeMediaKind === 'video' : ['image','webcam'].includes(activeMediaKind);
  if (!matching && activeMediaKind) document.dispatchEvent(new Event('workspace:reset'));
  if (!activeMediaKind) {
    previewEmptyTitle.textContent = isVideo ? 'Thêm video để bắt đầu' : 'Thêm hình ảnh để bắt đầu';
    previewEmptyDescription.textContent = isVideo ? 'Chọn tệp hoặc kéo thả video MP4, WebM, AVI. Sau khi nạp, nhấn Chạy.' : 'Kéo thả ảnh PNG, JPG hoặc chọn Webcam trực tiếp.';
  }
  if(location.hash !== '#'+view) history.pushState(null,'','#'+view);
}
window.addEventListener('hashchange',()=>navigateTo(location.hash.slice(1)));
document.querySelector('.brand').addEventListener('click',e=>{e.preventDefault();navigateTo('home');});
document.querySelectorAll('.home-actions a').forEach(link=>link.addEventListener('click',e=>{e.preventDefault();navigateTo(link.hash.slice(1));}));
// Validate the actual dropped file too, since accept only filters the file picker.
imageFileInput.addEventListener('change',e=>{
  const file=imageFileInput.files?.[0]; if(!file)return;
  const ext=file.name.split('.').pop().toLowerCase();
  const allowed=currentView==='video'?['mp4','webm','avi']:['png','jpg','jpeg'];
  if(!allowed.includes(ext)){
    e.stopImmediatePropagation();imageFileInput.value='';fileFeedback.hidden=false;fileFeedback.classList.remove('is-success');
    fileFeedback.textContent=currentView==='video'?'Hãy chọn video MP4, WebM hoặc AVI.':'Hãy chọn ảnh PNG/JPG. Để tải video, mở mục Video.';
  }
},true);
navigateTo(['home','video','recognition'].includes(location.hash.slice(1))?location.hash.slice(1):'home');
