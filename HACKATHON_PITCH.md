# 🎤 Cyber Samvad Hackathon 2026 — Jury Pitch Deck & Presentation Guide
### Problem Statement 1: "Is This Video Real?" — Indian Deepfake Detection Platform
**Project Name:** SatyaDrishti AI (सत्य दृष्टि)

---

## ⏱️ 3-Minute Presentation Pitch Script

### Slide 1: The Menace in India (0:00 - 0:40)
> *"Respected Jury members, India is the world's largest WhatsApp and smartphone market with over 900 million connected users. But this hyper-connectivity is being weaponized. Deepfake videos are no longer a theoretical risk—they are causing real damage:*
> 1. *Fabricated political speeches sparking communal tensions.*
> 2. *Fake celebrity endorsements duping middle-class citizens into investment scams.*
> 3. *And the horrific rise of 'Digital Arrest' scams where fraudsters impersonate police officers and CBI officials on video calls.*
>
> *The common citizen, local journalist, and small-town police cell have no accessible, trustworthy tool to verify these videos before panic spreads."*

---

### Slide 2: Introducing SatyaDrishti (0:40 - 1:15)
> *"To solve this national challenge, we built **SatyaDrishti AI (सत्य दृष्टि)**—India's first lightweight, multi-signal media forensics platform.*
>
> *Unlike complex laboratory software requiring costly GPUs, SatyaDrishti runs instantly in any modern web browser or smartphone. It delivers three unequivocal outcomes:*
> - 🔴 **'Possible manipulation detected'** *(with risk score & evidence clues)*
> - 🟢 **'No strong signs detected'** *(when authentic biomarkers pass)*
> - 🟡 **'Unable to assess'** *(a crucial ethical feature that refuses to make reckless guesses on heavily degraded footage)*
>
> *Crucially, we provide explanations in **7 Indian languages**—Hindi, Marathi, Gujarati, Bengali, Tamil, Telugu, and English—empowering citizens across Bharat."*

---

### Slide 3: How the Technology Works (1:15 - 2:00)
> *"Rather than relying on a single 'black-box' neural network, SatyaDrishti implements a multi-signal forensic pipeline:*
> 1. **Spatial Boundary Analysis:** *Measures Laplacian edge-gradient variance around the face to catch splicing seams from FaceSwap and DeepFaceLab.*
> 2. **2D FFT Spectral Decomposition:** *Identifies the high-frequency grid artifacts left by GAN decoders and diffusion upsampling.*
> 3. **Biological Dynamics:** *Tracks natural human eye-blink frequency (normal humans blink 12-20 times a minute). Synthetic avatars either never blink or have erratic flutter.*
> 4. **Chrominance Consistency:** *Measures skin-tone tone mismatch in YCrCb color space between the swapped face mask and the speaker's neck."*

---

### Slide 4: Live Demonstration Walkthrough (2:00 - 2:40)
> *(Clicking Demo 2 on screen)*
> *"Let's test an AI Face-Swap video. Within 2 seconds, SatyaDrishti flags it with **96% Manipulation Risk**. It points out the exact reason: abnormal eyelid inactivity and boundary gradient divergence at frame #42.*
>
> *(Clicking Demo 1)*
> *Now we test an authentic speaker: SatyaDrishti confirms **8% Risk — No strong signs detected**, noting natural blink rates and smooth contour blending.*
>
> *(Clicking Demo 4)*
> *And look at this 144p WhatsApp forward: instead of giving a wrong verdict, the platform responsibly reports **'Unable to assess'**, explaining that the video is too compressed to prove manipulation ethically."*

---

### Slide 5: Societal Impact & Next Steps (2:40 - 3:00)
> *"Every analysis includes a one-click printable **Forensic Audit Dossier** with a cryptographic SHA-256 hash for police reporting, and a direct link to the **1930 Cyber Fraud Helpline**.*
>
> *SatyaDrishti protects Indian elections, protects citizen savings, and brings scientific truth to the palm of every Indian. Thank you!"*

---

## 💡 Anticipated Jury Questions & Answers (Q&A Prep)

### Q1: "Why did you include an 'Unable to assess' verdict instead of always predicting Fake or Real?"
**Answer:**
> *"That is one of the most critical ethical guidelines in modern media forensics and the hackathon prompt. A heavily compressed 144p video from WhatsApp has lost most micro-expressions and high-frequency edge data. If an AI tool guesses on such video, it has a high false-positive rate, which could ruin an innocent person's reputation or accuse an authentic speaker of lying. When quality is too degraded, the only honest, scientifically valid verdict is 'Unable to assess'."*

### Q2: "Can your system detect whole-frame generative AI like OpenAI Sora or Runway Gen-3?"
**Answer:**
> *"Whole-frame diffusion models generate the entire scene simultaneously rather than splicing a face onto an existing body. For those, facial boundary seams don't exist. That is why our pipeline also includes 2D FFT spectral decay analysis (which detects diffusion frequency artifacts) and biological eye-blink tracking. In our future roadmap, we are integrating audio phoneme synchronization and C2PA Content Credentials to detect full-scene synthesis."*

### Q3: "How does this platform scale for millions of WhatsApp users in India?"
**Answer:**
> *"Because our core forensic algorithms (FFT, Haar, Laplacian, and chrominance tracking) are mathematically optimized in C++ via OpenCV and NumPy, each video analysis takes under 2 seconds on standard CPU cores—no expensive GPU servers required! This allows a high-throughput WhatsApp Bot deployment where community administrators can forward videos to a designated number and receive an instant verdict card."*

### Q4: "How do you ensure privacy and data security?"
**Answer:**
> *"All uploads are stored temporarily in isolated staging directories with cryptographic SHA-256 hashing. No personal facial data is ever harvested or trained on without consent, adhering strictly to India's Digital Personal Data Protection (DPDP) Act 2023."*
