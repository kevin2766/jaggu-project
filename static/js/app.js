// SatyaDrishti Client Application Logic - Cyber Samvad 2026
document.addEventListener('DOMContentLoaded', () => {
  let selectedFile = null;
  let selectedDemo = null;
  let currentAnalysis = null;
  let activeLang = 'en';

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const videoPlayer = document.getElementById('videoPlayer');
  const videoPlaceholder = document.getElementById('videoPlaceholder');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const demoButtons = document.querySelectorAll('.btn-demo');
  const scannerIndicator = document.getElementById('scannerIndicator');
  const scannerFill = document.getElementById('scannerFill');
  const scannerStatus = document.getElementById('scannerStatus');
  const resultsCard = document.getElementById('resultsCard');

  // File Upload Handlers
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-active');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-active');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-active');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    selectedDemo = null;
    demoButtons.forEach(b => b.classList.remove('active'));

    document.getElementById('dropzoneTitle').innerText = file.name;
    document.getElementById('dropzoneSub').innerText = `${(file.size / (1024 * 1024)).toFixed(2)} MB - Ready to analyze`;

    const blobUrl = URL.createObjectURL(file);
    loadVideoSource(blobUrl);
    btnAnalyze.disabled = false;
  }

  // Demo Buttons Selection
  demoButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      demoButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      selectedDemo = btn.dataset.demo;
      selectedFile = null;
      fileInput.value = '';

      document.getElementById('dropzoneTitle').innerText = 'Drop video here or click to browse';
      document.getElementById('dropzoneSub').innerText = `Demo selected: ${btn.querySelector('.demo-title').innerText}`;

      const demoUrl = `/static/demo_samples/demo_${selectedDemo}.mp4`;
      loadVideoSource(demoUrl);
      btnAnalyze.disabled = false;
    });
  });

  function loadVideoSource(url) {
    videoPlaceholder.style.display = 'none';
    videoPlayer.style.display = 'block';
    videoPlayer.src = url;
    videoPlayer.load();
  }

  // Analyze Action
  btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile && !selectedDemo) return;

    btnAnalyze.disabled = true;
    scannerIndicator.style.display = 'block';
    scannerFill.style.width = '10%';
    scannerStatus.innerText = 'Extracting spatial frames and facial contours...';

    const formData = new FormData();
    if (selectedDemo) {
      formData.append('demo_choice', selectedDemo);
    } else {
      formData.append('video', selectedFile);
    }

    // Animate progress simulation
    let progress = 10;
    const interval = setInterval(() => {
      progress += 12;
      if (progress < 90) {
        scannerFill.style.width = `${progress}%`;
        if (progress > 30 && progress < 60) {
          scannerStatus.innerText = 'Computing 2D FFT spectral decay and boundary gradients...';
        } else if (progress >= 60) {
          scannerStatus.innerText = 'Measuring biological eye-blink rate and audio-motion coherence...';
        }
      }
    }, 400);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      clearInterval(interval);
      scannerFill.style.width = '100%';
      scannerStatus.innerText = 'Forensic synthesis complete!';

      const data = await response.json();
      setTimeout(() => {
        scannerIndicator.style.display = 'none';
        btnAnalyze.disabled = false;
        renderResults(data);
      }, 500);

    } catch (err) {
      clearInterval(interval);
      scannerIndicator.style.display = 'none';
      btnAnalyze.disabled = false;
      alert('Error during analysis: ' + err.message);
    }
  });

  function renderResults(data) {
    currentAnalysis = data;
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });

    // Verdict Banner
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictDesc = document.getElementById('verdictDesc');
    const riskScoreVal = document.getElementById('riskScoreVal');
    const confScoreVal = document.getElementById('confScoreVal');

    verdictBanner.className = 'verdict-banner';

    if (data.verdict === 'Possible manipulation detected') {
      verdictBanner.classList.add('verdict-fake');
      verdictTitle.innerHTML = '⚠️ Possible manipulation detected';
    } else if (data.verdict === 'No strong signs detected') {
      verdictBanner.classList.add('verdict-real');
      verdictTitle.innerHTML = '🛡️ No strong signs detected';
    } else {
      verdictBanner.classList.add('verdict-unable');
      verdictTitle.innerHTML = '❓ Unable to assess';
    }

    verdictDesc.innerText = data.summary;
    riskScoreVal.innerText = `${data.risk_score}%`;
    confScoreVal.innerText = `${data.confidence_score}%`;

    // Multilingual Explanation
    updateMultilingualExplanation(activeLang);

    // Telemetry Gauges
    const m = data.metrics;
    document.getElementById('metricSeam').innerText = m.spatial_seam_index;
    document.getElementById('metricSeamStatus').innerText = m.spatial_seam_index > 0.32 ? 'High Discrepancy' : 'Normal Gradient';
    document.getElementById('metricSeamStatus').className = 'telemetry-status ' + (m.spatial_seam_index > 0.32 ? 'status-anom' : 'status-normal');

    document.getElementById('metricFFT').innerText = m.spectral_fft_ratio;
    document.getElementById('metricFFTStatus').innerText = m.spectral_fft_ratio > 0.8 ? 'Synthetic Grid Spike' : 'Natural Harmonics';
    document.getElementById('metricFFTStatus').className = 'telemetry-status ' + (m.spectral_fft_ratio > 0.8 ? 'status-anom' : 'status-normal');

    document.getElementById('metricBlink').innerText = `${m.blinks_detected} (${m.blink_rate_cpm}/m)`;
    const blinkNormal = m.blink_rate_cpm >= 10 && m.blink_rate_cpm <= 50;
    document.getElementById('metricBlinkStatus').innerText = blinkNormal ? 'Normal Biological Range' : 'Abnormal Rate / Stare';
    document.getElementById('metricBlinkStatus').className = 'telemetry-status ' + (blinkNormal ? 'status-normal' : 'status-anom');

    document.getElementById('metricChroma').innerText = m.chroma_shift_delta;
    document.getElementById('metricJitter').innerText = `${m.motion_jitter_index} px`;
    document.getElementById('metricSharpness').innerText = data.metadata.sharpness_score;

    // Clues List
    const cluesList = document.getElementById('cluesList');
    cluesList.innerHTML = '';
    if (data.clues && data.clues.length > 0) {
      data.clues.forEach(clue => {
        const item = document.createElement('div');
        const sevClass = clue.severity ? clue.severity.toLowerCase() : 'info';
        item.className = `clue-item ${sevClass}`;
        item.innerHTML = `
          <div class="clue-content">
            <h4>${clue.name}</h4>
            <p>${clue.description}</p>
          </div>
          <span class="clue-timestamp">${clue.timestamp}</span>
        `;
        cluesList.appendChild(item);
      });
    } else {
      cluesList.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No anomalous cues identified.</p>';
    }

    // Keyframes Gallery
    const gallery = document.getElementById('keyframesGallery');
    gallery.innerHTML = '';
    if (data.keyframe_evidence && data.keyframe_evidence.length > 0) {
      data.keyframe_evidence.forEach(kf => {
        const box = document.createElement('div');
        box.className = 'keyframe-box';
        box.innerHTML = `
          <img src="${kf.thumbnail}" alt="Frame ${kf.frame_idx}" />
          <div class="keyframe-meta">
            <span>#${kf.frame_idx}</span>
            <span style="color: ${kf.is_suspicious ? '#ef4444' : '#10b981'}">${kf.timestamp}</span>
          </div>
        `;
        box.addEventListener('click', () => {
          // Jump video to this timestamp
          const sec = parseFloat(kf.timestamp.replace('s', ''));
          if (!isNaN(sec)) {
            videoPlayer.currentTime = sec;
            videoPlayer.play();
          }
        });
        gallery.appendChild(box);
      });
    } else {
      gallery.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No keyframe thumbnails available.</p>';
    }
  }

  // Multilingual Selector Tabs
  const langTabs = document.querySelectorAll('.lang-tab');
  langTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      langTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      activeLang = tab.dataset.lang;
      updateMultilingualExplanation(activeLang);
    });
  });

  function updateMultilingualExplanation(lang) {
    if (!currentAnalysis || !currentAnalysis.multilingual) return;
    const multi = currentAnalysis.multilingual[lang] || currentAnalysis.multilingual['en'];

    document.getElementById('plainBadge').innerText = multi.badge;
    document.getElementById('plainHeadline').innerText = multi.headline;
    document.getElementById('plainText').innerText = multi.explanation;
    document.getElementById('plainAction').innerText = multi.action;
  }

  // Print Dossier Action
  const btnPrint = document.getElementById('btnPrintDossier');
  if (btnPrint) {
    btnPrint.addEventListener('click', () => window.print());
  }

  // Pitch Deck Modal Presentation
  const btnPitch = document.getElementById('btnPitch');
  const modalPitch = document.getElementById('modalPitch');
  const btnCloseModal = document.getElementById('btnCloseModal');
  const slides = document.querySelectorAll('.pitch-slide');
  const btnPrevSlide = document.getElementById('btnPrevSlide');
  const btnNextSlide = document.getElementById('btnNextSlide');
  const slideCounter = document.getElementById('slideCounter');
  let currentSlide = 0;

  function showSlide(index) {
    slides.forEach((s, idx) => s.classList.toggle('active', idx === index));
    slideCounter.innerText = `Slide ${index + 1} of ${slides.length}`;
    btnPrevSlide.disabled = index === 0;
    btnNextSlide.disabled = index === slides.length - 1;
  }

  if (btnPitch) {
    btnPitch.addEventListener('click', () => {
      currentSlide = 0;
      showSlide(currentSlide);
      modalPitch.classList.add('active');
    });
  }

  if (btnCloseModal) {
    btnCloseModal.addEventListener('click', () => modalPitch.classList.remove('active'));
  }

  modalPitch.addEventListener('click', (e) => {
    if (e.target === modalPitch) modalPitch.classList.remove('active');
  });

  if (btnPrevSlide) {
    btnPrevSlide.addEventListener('click', () => {
      if (currentSlide > 0) {
        currentSlide--;
        showSlide(currentSlide);
      }
    });
  }

  if (btnNextSlide) {
    btnNextSlide.addEventListener('click', () => {
      if (currentSlide < slides.length - 1) {
        currentSlide++;
        showSlide(currentSlide);
      }
    });
  }

  document.addEventListener('keydown', (e) => {
    if (!modalPitch.classList.contains('active')) return;
    if (e.key === 'ArrowRight' && currentSlide < slides.length - 1) {
      currentSlide++;
      showSlide(currentSlide);
    } else if (e.key === 'ArrowLeft' && currentSlide > 0) {
      currentSlide--;
      showSlide(currentSlide);
    } else if (e.key === 'Escape') {
      modalPitch.classList.remove('active');
    }
  });
});
