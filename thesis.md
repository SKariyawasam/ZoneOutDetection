# DipSEER: A Multimodal Cognitive Focus Detection System Using SVD-Based Visual Analysis, Multiresolution CNN-LSTM Physiological Inference, and Dynamic Signal-Quality Adaptive Fusion

**Student Name:** Shehan Kariyawasam
**Student ID:** CB013130
**Degree Programme:** BSc (Hons) Computer Science
**Supervisor:** [Supervisor Name]

---

## ABSTRACT

Mind wandering — the phenomenon of attentional decoupling from an ongoing task toward task-unrelated thought — is a pervasive cognitive challenge affecting academic learning outcomes, occupational productivity, and road safety. Despite growing interest in automated detection, existing approaches predominantly rely on single-modality sensing (either facial video analysis or electroencephalography) and employ static fusion architectures that cannot adapt to real-world signal degradation. This thesis presents **DipSEER**, a novel multimodal cognitive focus detection system that integrates webcam-based Singular Value Decomposition (SVD) visual analysis with Samsung Galaxy Watch physiological telemetry (photoplethysmographic heart rate and inertial measurement unit rotation vectors) through a dynamic signal-quality adaptive late fusion framework. The visual processing pipeline (Module A) performs real-time 8×8 ocular energy grid decomposition, SVD-based frequency-selective singular vector extraction, TSFEL statistical feature engineering, and XGBoost classification, enriched with Bosch and D'Mello's (2021) head dynamics velocity analysis and blank stare fixation indexing. The physiological processing pipeline (Module B) employs a Multimodal Multiresolution Dual-Pooling CNN-LSTM architecture inspired by John et al. (2021) and Najafi et al. (2023), processing heart rate variability indices (SDNN, RMSSD) and IMU-derived wrist motion stability across temporally mismatched sampling rates. The decision-level fusion engine (Module C) implements Wei et al.'s (2018) dynamic signal-quality weighted fusion strategy, where modality weights are continuously recalculated based on real-time visual tracking quality and physiological signal integrity. Closed-loop interventions are delivered simultaneously through native Windows desktop notifications and Galaxy Watch haptic vibration alerts. Trained on the DAiSEE dataset (9,068 video clips, 112 subjects) and the DipSeer Dataset (65 recording sessions, 21 subjects), the system achieves real-time 1 Hz inference with sub-3ms forward pass latency on CPU. The prototype demonstrates robust degradation handling across four operational scenarios: normal desk study, typing-induced wrist motion, head turning/dim lighting, and complete watch disconnection.

**Keywords:** Mind wandering detection, multimodal fusion, SVD, CNN-LSTM, heart rate variability, wearable sensing, attention monitoring, signal-quality adaptive fusion

---

## LIST OF TABLES

| Table | Title | Page |
|-------|-------|------|
| Table 2.1 | Stakeholder Viewpoints Analysis | 10 |
| Table 2.2 | Survey Demographic Distribution | 16 |
| Table 2.3 | Survey Results: Mind Wandering Frequency | 17 |
| Table 2.4 | Survey Results: Willingness to Use Monitoring Technology | 18 |
| Table 2.5 | Summary of Requirement Elicitation Findings | 22 |
| Table 2.6 | Functional Requirements Specification | 27 |
| Table 2.7 | Non-Functional Requirements Specification | 28 |
| Table 3.1 | Design Goals and Rationale | 29 |
| Table 4.1 | Technology Stack Summary | 39 |
| Table 4.2 | Dataset Characteristics Comparison | 37 |
| Table 5.1 | Dynamic Fusion Weight Adaptation Scenarios | 44 |

---

## LIST OF FIGURES

| Figure | Title | Page |
|--------|-------|------|
| Figure 2.1 | Rich Picture Diagram | 9 |
| Figure 2.2 | Stakeholder Onion Model | 10 |
| Figure 2.3 | Context Diagram | 23 |
| Figure 2.4 | Use Case Diagram | 24 |
| Figure 3.1 | High-Level Architecture Diagram | 30 |
| Figure 3.2 | Component Diagram | 32 |
| Figure 3.3 | Data Flow Diagram | 33 |
| Figure 3.4 | System Process Activity Diagram | 34 |
| Figure 3.5 | Web Dashboard UI Design | 35 |
| Figure 4.1 | SVD Ocular Energy Grid Decomposition Pipeline | 40 |
| Figure 4.2 | Multiresolution Dual-Pooling CNN-LSTM Architecture | 41 |
| Figure 4.3 | Dynamic Signal-Quality Adaptive Fusion Engine | 42 |
| Figure 5.1 | Live Dashboard with Dynamic Weight Badges | 45 |

---

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|---|---|
| API | Application Programming Interface |
| BLE | Bluetooth Low Energy |
| CNN | Convolutional Neural Network |
| CPU | Central Processing Unit |
| CSV | Comma-Separated Values |
| DAiSEE | Dataset for Affective States in E-Environments |
| EAR | Eye Aspect Ratio |
| EDA | Electrodermal Activity |
| EEG | Electroencephalography |
| FFT | Fast Fourier Transform |
| FPS | Frames Per Second |
| GPU | Graphics Processing Unit |
| HR | Heart Rate |
| HRV | Heart Rate Variability |
| IMU | Inertial Measurement Unit |
| JSON | JavaScript Object Notation |
| LSTM | Long Short-Term Memory |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| PPG | Photoplethysmography |
| RMSSD | Root Mean Square of Successive Differences |
| SDNN | Standard Deviation of NN Intervals |
| SVD | Singular Value Decomposition |
| TLS | Transport Layer Security |
| TSFEL | Time Series Feature Extraction Library |
| UDP | User Datagram Protocol |
| UI | User Interface |
| WSS | WebSocket Secure |

---

## 1. INTRODUCTION

### 1.1 Chapter Overview

This chapter introduces the problem domain of mind wandering detection in educational and occupational settings, articulates the specific problem definition addressed by this research, outlines the aims and objectives, identifies the research gap and novelty, and summarises the contributions to both the problem and research domains.

### 1.2 Problem Domain

Mind wandering, formally defined as the redirection of attention away from an ongoing primary task toward self-generated, task-unrelated thoughts (Smallwood and Schooler, 2015), represents one of the most pervasive yet underexplored cognitive phenomena affecting human performance. Research consistently demonstrates that individuals spend approximately 25–50% of their waking hours engaged in mind wandering episodes (Killingsworth and Gilbert, 2010), with significant implications across multiple domains.

In educational contexts, mind wandering during lectures and self-study sessions has been empirically linked to reduced comprehension, impaired information encoding, and degraded examination performance (Risko et al., 2012). Studies utilising experience sampling methodologies have found that university students report mind wandering during 20–40% of lecture time (Wammes, Boucher and Seli, 2016), with the frequency increasing as lecture duration extends. The phenomenon is particularly pronounced in online and distance learning environments, where the absence of social accountability mechanisms and the presence of digital distractions compound the underlying attentional challenges (D'Mello, 2016).

From a physiological perspective, mind wandering episodes are accompanied by measurable changes in autonomic nervous system activity. Research by Smallwood et al. (2004) demonstrated that heart rate variability indices, particularly the root mean square of successive differences (RMSSD) in R-R intervals, exhibit statistically significant modulations during attentional lapses compared to sustained focus periods. Similarly, studies utilising wearable inertial measurement units have revealed characteristic postural and kinematic signatures associated with disengagement, including reduced head movement variability and increased stillness in gaze patterns (Bosch and D'Mello, 2021).

The convergence of affordable consumer wearable technology — including smartwatches equipped with optical photoplethysmographic (PPG) heart rate sensors and three-axis rotation vector sensors — with advances in deep learning architectures for multimodal time-series fusion creates an unprecedented opportunity for developing real-time, non-invasive mind wandering detection systems that operate outside controlled laboratory environments (Hickey et al., 2021; Can, Chalabianloo and Ersoy, 2019). This research occupies the intersection of affective computing, wearable biosensing, computer vision, and human-computer interaction, domains that have historically evolved in relative isolation despite their complementary contributions to understanding human cognitive states (Giannakos et al., 2022).

### 1.3 Problem Definition

Current approaches to mind wandering detection suffer from three fundamental limitations that constrain their practical deployment:

**Single-modality dependency.** The majority of existing systems rely exclusively on either facial video analysis (Lim, Lee and Kim, 2020; Bosch and D'Mello, 2021) or electroencephalographic (EEG) brain-computer interfaces (Baldwin et al., 2017; Dong, Hu and Liu, 2021). Video-only systems are vulnerable to occlusion, illumination changes, and head pose variations, while EEG-based approaches require intrusive sensor placement that precludes naturalistic use in classrooms and workplaces.

**Static fusion architectures.** Systems that do combine multiple sensing modalities typically employ fixed-weight fusion strategies where the relative contribution of each modality is predetermined during training and remains constant during inference (Sharma et al., 2019). This approach fails catastrophically when one modality experiences signal degradation — for example, when a webcam loses face tracking due to temporary occlusion, a fixed-weight system inappropriately continues to integrate degraded visual features, contaminating the overall classification with noise rather than gracefully redistributing weight to the physiological channel.

**Absence of closed-loop intervention.** Most detection systems function as passive monitoring tools that log cognitive state data for post-hoc analysis, without providing real-time feedback or corrective nudges to the user (D'Mello, 2016). This limits their practical utility, as the value of detecting mind wandering lies not merely in its identification but in the timely re-engagement of the user's attention.

This project therefore addresses the following composite problem: *How can webcam-based facial analysis and consumer smartwatch physiological telemetry be fused through a dynamic, signal-quality adaptive architecture to detect mind wandering in real time, while providing simultaneous closed-loop multimodal interventions?*

### 1.4 Aims and Objectives

#### 1.4.1 Aims

The primary aim of this project is to design, implement, and evaluate a multimodal cognitive focus detection prototype — designated **DipSEER** — that integrates visual and physiological sensing modalities through a dynamic signal-quality adaptive fusion framework, capable of real-time mind wandering detection and closed-loop multi-device intervention delivery.

#### 1.4.2 Research Objectives

The following research objectives decompose the primary aim into concrete, measurable deliverables:

**RO1.** To develop a real-time visual processing pipeline (Module A) that extracts cognitive engagement indicators from webcam video using Singular Value Decomposition (SVD) of ocular energy distributions, Time Series Feature Extraction Library (TSFEL) statistical features, and XGBoost classification, enriched with temporal head dynamics and blank stare fixation indices derived from Bosch and D'Mello's (2021) computational framework.

**RO2.** To design and train a Multimodal Multiresolution Dual-Pooling CNN-LSTM deep neural network (Module B) that processes physiological time-series data from the Samsung Galaxy Watch — including heart rate variability metrics (SDNN, RMSSD) and IMU rotation vector stability — across temporally mismatched sampling rates, employing the architectural principles established by John et al. (2021) and Najafi et al. (2023).

**RO3.** To implement a Dynamic Signal-Quality Adaptive Fusion engine (Module C) based on Wei et al.'s (2018) weighted multichannel fusion strategy, where modality weights are continuously recalculated in response to real-time signal quality indicators derived from face tracking stability (visual quality) and wrist motion intensity (physiological quality).

**RO4.** To develop a bidirectional communication infrastructure between the laptop backend, Samsung Galaxy Watch, and web monitoring dashboard, incorporating zero-configuration UDP auto-discovery, TLS-encrypted WebSocket Secure (WSS) streaming, and native platform intervention mechanisms (Windows desktop notifications and Wear OS haptic vibration alerts).

**RO5.** To evaluate the integrated system using the DAiSEE dataset (9,068 video clips, 112 subjects) for visual model training and the DipSeer Dataset (65 recording sessions, 21 subjects with synchronised smartwatch telemetry) for the multiresolution physiological model, demonstrating the system's ability to adapt fusion weights dynamically across degraded-signal scenarios.

### 1.5 Novelty of Research

#### 1.5.1 Problem Novelty

While mind wandering detection has been investigated using individual modalities — including EEG (Dong, Hu and Liu, 2021; Grandchamp, Braboszcz and Delorme, 2014), facial video (Bosch and D'Mello, 2021; Lim, Lee and Kim, 2020), and eye tracking (Bixler and D'Mello, 2016; Faber, Bixler and D'Mello, 2018) — no existing system combines consumer-grade webcam SVD-based visual analysis with smartwatch PPG and IMU telemetry for mind wandering detection. The majority of wearable-augmented cognitive monitoring systems target stress detection (Can, Chalabianloo and Ersoy, 2019; Hickey et al., 2021) or emotion recognition (Wei et al., 2018), rather than the specific phenomenon of attentional disengagement during knowledge work.

#### 1.5.2 Solution Novelty

The solution introduces three novel technical contributions:

First, the integration of SVD-based ocular energy decomposition with real-time TSFEL temporal feature extraction and head dynamics/blank stare indexing within a unified visual processing pipeline has not been previously demonstrated for real-time mind wandering detection.

Second, the application of dual-pooling (average + maximum temporal pooling) multiresolution CNN-LSTM architectures — originally proposed for IoT wearable activity recognition (John et al., 2021; Najafi et al., 2023) — to the domain of cognitive focus estimation from PPG and IMU signals represents a novel cross-domain transfer.

Third, the dynamic signal-quality adaptive fusion engine that continuously modulates modality weights based on real-time noise estimation (face tracking quality and wrist motion intensity) addresses a fundamental limitation of existing multimodal systems that employ static fusion coefficients.

### 1.6 Research Gap

The literature review reveals a significant gap at the intersection of three research areas:

Multimodal mind wandering detection systems that combine visual and physiological modalities are exceedingly rare. Existing multimodal approaches primarily target emotion recognition (Wei et al., 2018; Sharma et al., 2019) or cognitive load estimation (Gjoreski et al., 2020), rather than the binary or graded detection of attentional disengagement. Furthermore, among the systems that do employ sensor fusion, none implement dynamic signal-quality adaptive weighting that responds to real-time sensor degradation — a critical requirement for deployment outside controlled laboratory environments where camera occlusion, variable lighting, and wrist movement artefacts are commonplace.

The research gap can therefore be characterised as the absence of a multimodal mind wandering detection system that (a) fuses webcam-based visual analysis with consumer smartwatch physiological telemetry, (b) employs dynamic signal-quality adaptive fusion weights, and (c) delivers closed-loop multi-device interventions.

### 1.7 Contribution to the Body of Knowledge

#### 1.7.1 Contribution to the Problem Domain

This research contributes a functional prototype that enables educators, self-directed learners, and knowledge workers to receive real-time feedback on their attentional state using only a standard webcam and a consumer smartwatch — devices they already possess — without requiring specialised neuroimaging hardware or intrusive sensor placement.

#### 1.7.2 Contribution to the Research Domain

The research contributes a reference implementation and evaluation of dynamic signal-quality adaptive multimodal fusion for cognitive state classification, demonstrating that real-time noise-responsive weight modulation outperforms static fusion in degraded-signal scenarios. Additionally, the cross-domain application of multiresolution CNN-LSTM architectures from IoT activity recognition to cognitive focus estimation expands the known utility of these architectures.

### 1.8 Research Challenge

The primary research challenges encountered during development include:

**Temporal resolution mismatch.** The webcam captures visual features at 20 frames per second, while the Galaxy Watch's PPG sensor reports heart rate at approximately 1 Hz. Fusing modalities with a 20:1 sampling rate ratio requires architectural solutions that preserve temporal granularity without introducing artificial interpolation artefacts. The multiresolution CNN-LSTM architecture addresses this by processing each modality through dedicated convolutional branches before cross-resolution concatenation.

**Real-time computational constraints.** The system must achieve sub-second inference latency on a standard laptop CPU without dedicated GPU acceleration, as DirectML TensorFlow on Windows encounters kernel registration failures for standard Keras LSTM operations on GPU. All deep learning inference is therefore pinned to CPU execution, necessitating lightweight model architectures.

**Zero-configuration networking.** The smartwatch and laptop must establish secure bidirectional communication without manual IP configuration, even as the laptop transitions between Wi-Fi networks. The UDP subnet broadcast auto-discovery mechanism and SharedPreferences-based IP persistence address this challenge.

### 1.9 Chapter Summary

This chapter has established the problem domain of mind wandering detection, articulated the specific problem of single-modality dependency, static fusion, and the absence of closed-loop intervention. It has outlined five research objectives, identified the research gap at the intersection of multimodal fusion, consumer wearable sensing, and cognitive state classification, and characterised the primary technical challenges. The following chapter presents the software requirement specification derived from stakeholder analysis and empirical requirement elicitation.

---

## 2. SOFTWARE REQUIREMENT SPECIFICATION

### 2.1 Chapter Overview

This chapter presents the systematic elicitation, analysis, and documentation of software requirements for the DipSEER system. It employs four complementary requirement elicitation methodologies — literature review, survey, interview, and prototyping — to triangulate stakeholder needs, and documents the findings through a rich picture diagram, stakeholder analysis, context diagram, use case specifications, and a structured requirements catalogue.

### 2.2 Rich Picture Diagram

The rich picture diagram captures the complex sociotechnical landscape surrounding cognitive focus monitoring in educational environments. The central scenario depicts a university student engaged in a self-study session at a desk, wearing a Samsung Galaxy Watch on their wrist and sitting before a laptop with an integrated webcam. The diagram illustrates the tension between three competing forces: the student's desire for uninterrupted productivity, the pervasive threat of digital distractions (social media notifications, entertainment content), and the latent physiological signals (heart rate fluctuations, postural changes) that accompany attentional disengagement.

Key actors identified through the rich picture include the student user, academic supervisors, educational technology developers, lecturers monitoring class engagement, and privacy-conscious observers concerned about continuous biometric surveillance. The diagram highlights the emotional associations with cognitive monitoring — students express both interest in self-awareness tools and anxiety about being watched — motivating design decisions around data locality (all processing occurs on the student's own laptop without cloud transmission) and transparent feedback mechanisms.

### 2.3 Stakeholder Analysis

#### 2.3.1 Stakeholder Onion Model

The stakeholder onion model organises identified stakeholders across four concentric layers of interaction intensity:

**Core (The System):** The DipSEER software system itself, comprising the laptop backend inference engine, the Galaxy Watch foreground service, and the web monitoring dashboard.

**First Layer (Direct Users):** University students and self-directed learners who wear the smartwatch and sit before the webcam during study sessions. These stakeholders interact directly with the system and receive real-time feedback through visual and haptic interventions.

**Second Layer (Indirect Beneficiaries):** Lecturers and academic supervisors who benefit from aggregated engagement insights (though the current prototype does not implement data export), educational psychologists researching mind wandering interventions, and accessibility specialists who may adapt the system for neurodivergent populations.

**Third Layer (External Stakeholders):** Institutional ethics review boards concerned with biometric data governance, consumer electronics manufacturers (Samsung, Google) whose platforms host the wearable application, and the broader human-computer interaction research community.

#### 2.3.2 Stakeholder Viewpoints

| Stakeholder | Primary Concern | Key Requirement | Priority |
|---|---|---|---|
| University Student | Maintaining study focus without intrusive monitoring | Non-invasive sensing; clear, actionable alerts; data privacy | Critical |
| Self-Directed Learner | Self-awareness of attention patterns during online learning | Real-time visual dashboard with engagement metrics | High |
| Lecturer | Understanding class-wide engagement trends | Scalable multi-user monitoring (future scope) | Medium |
| Educational Psychologist | Validated cognitive state classification accuracy | Ground truth alignment with established attention measures | High |
| Privacy Advocate | Protection of biometric facial and cardiac data | Local-only processing; no cloud data transmission; encrypted channels | Critical |
| Wearable Platform Developer | Compliance with Wear OS battery and sensor API constraints | Efficient sensor sampling; proper foreground service lifecycle | Medium |
| Accessibility Specialist | Adaptation for sensory-impaired or neurodivergent users | Configurable alert modalities; adjustable sensitivity thresholds | Low |

### 2.4 Selection of Requirement Elicitation Methodologies

Four complementary elicitation methodologies were selected based on their suitability for the project's dual research-engineering nature:

**Literature Review.** A systematic examination of 35 academic publications spanning mind wandering detection, wearable biosensing, multimodal fusion, and affective computing was conducted to identify established functional patterns, technological constraints, and domain-specific requirements. This methodology is particularly appropriate for research-intensive projects where established scientific findings must inform system design decisions (Sommerville, 2015).

**Survey.** An online questionnaire was administered to 47 undergraduate and postgraduate students at the University to quantify the prevalence of self-reported mind wandering experiences, attitudes toward wearable-based monitoring, and preferences for intervention modality (visual, auditory, haptic). The survey employed a combination of Likert-scale items, multiple-choice questions, and open-ended responses.

**Interview.** Semi-structured interviews were conducted with three participants: an educational psychologist specialising in attentional regulation, a computer science lecturer with experience teaching large cohorts, and a student representative who had previously used productivity monitoring applications. Interviews were transcribed and subjected to thematic analysis.

**Prototyping.** An iterative prototyping approach was adopted whereby early system prototypes were demonstrated to potential users and their feedback was incorporated into subsequent design iterations. This methodology is well-suited to projects involving novel interface paradigms (such as real-time cognitive state visualisation) where users cannot articulate requirements without experiencing the system firsthand (Sommerville, 2015).

### 2.5 Discussion of Findings

#### 2.5.1 Literature Review

The literature review examined 35 publications drawn from IEEE Transactions on Affective Computing, Computational Intelligence and Neuroscience, Sensors, IEEE Transactions on Biomedical Circuits and Systems, and related venues. The key findings are organised thematically:

**Mind wandering detection modalities.** The literature identifies five primary sensing modalities for mind wandering detection: electroencephalography (EEG) (Baldwin et al., 2017; Dong, Hu and Liu, 2021; Grandchamp, Braboszcz and Delorme, 2014), webcam-based facial video analysis (Bosch and D'Mello, 2021; Stewart, Cardiel and Guo, 2020; Lim, Lee and Kim, 2020), dedicated eye tracking hardware (Bixler and D'Mello, 2016; Faber, Bixler and D'Mello, 2018; Hutt et al., 2017), electrodermal activity (EDA) (D'Mello, Mills and Bixler, 2021), and wearable photoplethysmographic heart rate monitoring (Can, Chalabianloo and Ersoy, 2019). Of these, EEG provides the highest temporal resolution and neural specificity but requires intrusive sensor placement incompatible with naturalistic settings. Webcam-based approaches offer non-contact sensing but are vulnerable to environmental variations. Wearable PPG monitoring provides continuous physiological indicators but cannot alone distinguish cognitive states without complementary contextual information.

**Multimodal fusion architectures.** Three fusion paradigms are identified in the literature: early fusion (feature-level concatenation), intermediate fusion (cross-modal attention mechanisms within deep architectures), and late fusion (decision-level integration of independently computed modality scores). Wei et al. (2018) demonstrate that weighted late fusion with signal-quality adaptive coefficients outperforms both fixed-weight late fusion and early concatenation for emotion recognition from multichannel physiological signals. John et al. (2021) introduce multiresolution data fusion using dedicated convolutional branches for modalities sampled at different temporal granularities, achieving superior performance over single-resolution architectures on IoT wearable activity recognition benchmarks. Najafi et al. (2023) augment convolutional feature extraction with dual temporal pooling — parallel average and maximum pooling — to simultaneously capture sustained trends and transient spikes in physiological time-series data.

**SVD-based visual feature extraction.** Singular Value Decomposition of facial sub-region energy matrices has been demonstrated as an effective alternative to deep CNN-based visual feature extraction for mind wandering prediction from online learning videos (Bhatia, Mitra and Gupta, 2021). This approach offers computational efficiency suitable for real-time inference on CPU, avoiding the GPU memory requirements associated with fine-tuning large pretrained vision models such as EfficientNet or Vision Transformers.

**Behavioural indicators of mind wandering.** Bosch and D'Mello (2021) identify three key facial and postural indicators that discriminate mind wandering from sustained attention: (a) reduced micro-saccadic variation in gaze direction (the "blank stare" phenomenon), (b) decreased head movement variability (postural freezing), and (c) increased eye closure duration. These indicators complement the SVD energy-based features by introducing temporal dynamics that capture the phenomenological characteristics of attentional lapses over multi-second time windows.

**Intervention design.** The literature on attention regulation interventions emphasises the importance of subtle, non-disruptive alerts that avoid startling the user or inducing test anxiety (D'Mello, 2016). Haptic feedback delivered through wrist-worn wearables is identified as a promising intervention modality due to its private, unobtrusive nature — perceptible only to the wearer without disturbing nearby individuals in shared study environments (Pielot, Church and de Oliveira, 2014).

#### 2.5.2 Survey

An online survey was administered to 47 university students (31 undergraduate, 16 postgraduate) across Computer Science, Engineering, and Social Sciences programmes. The survey instrument comprised 22 questions distributed across four sections: demographic information, self-reported mind wandering experiences, attitudes toward cognitive monitoring technology, and intervention modality preferences.

**Demographics.** Respondents were aged 18–30 (mean age 22.3), with 57% male, 40% female, and 3% non-binary/prefer not to say. Sixty-eight percent reported using a smartwatch or fitness tracker regularly, with Samsung Galaxy and Apple Watch comprising the dominant brands.

**Mind wandering frequency.** When asked "How often do you catch yourself mind wandering during lectures or study sessions?", 72% of respondents selected "Frequently" or "Very Frequently" on a 5-point Likert scale. Only 4% reported "Rarely" or "Never". These findings align with meta-analytic estimates reported in the attention research literature (Wammes, Boucher and Seli, 2016).

**Willingness to use monitoring technology.** Fifty-nine percent of respondents indicated they would "Definitely" or "Probably" use a non-invasive mind wandering detection tool during study sessions, provided that (a) no data leaves their personal device, (b) the system does not record or store video footage, and (c) alerts are subtle rather than alarming. Twenty-three percent were undecided, and 18% expressed reluctance due to privacy concerns or scepticism about detection accuracy.

**Preferred intervention modality.** Respondents were asked to rank four intervention modalities (visual dashboard notification, auditory chime, smartwatch vibration, and screen dimming) by preference. Smartwatch vibration ranked highest (selected as first preference by 41% of respondents), followed by visual dashboard notification (29%), auditory chime (19%), and screen dimming (11%). The preference for haptic intervention supports the decision to implement Galaxy Watch vibration as the primary re-engagement mechanism.

#### 2.5.3 Interview

Three semi-structured interviews were conducted and analysed using thematic analysis (Braun and Clarke, 2006). The following themes emerged:

**Theme 1: Self-awareness as primary motivation.** All three interviewees emphasised that the value of a mind wandering detection system lies primarily in enabling self-awareness rather than external surveillance. The educational psychologist noted: "Students who become aware of their attentional patterns can develop metacognitive strategies for self-regulation — this is fundamentally different from being monitored by someone else."

**Theme 2: Privacy as prerequisite.** The student representative expressed strong opinions about data locality: "If my face is being recorded and sent to a server, I would never use it. But if it's just my own laptop processing the video in real time without saving anything, that's completely different."

**Theme 3: Adaptive sensitivity.** The lecturer suggested that detection sensitivity should adapt to the user's baseline: "Some students fidget more than others while remaining fully engaged — the system needs to learn what's normal for each individual before flagging anomalies." This insight informed the design of the personalized calibration baseline in Module B's physiological inference pipeline.

#### 2.5.4 Prototyping

Early prototypes were demonstrated to five users in informal usability sessions. Key feedback included:

- The initial full-screen red alert overlay was perceived as overly aggressive and anxiety-inducing. This led to the addition of a dismissable "I'm Refocused" button and an 8-second suppression window after dismissal.
- Users requested visual confirmation that the camera and watch were actively tracking (not just connected), leading to the implementation of differentiated status indicators: "Tracking Face (Active)", "No Face Detected", "Watch Disconnected", "Watch Connected (Off-Body / Acquiring Pulse)", and "Tracking Physiological Telemetry (Active)".
- The double-pulse haptic pattern (300ms buzz, 150ms pause, 300ms buzz) was preferred over a single continuous vibration, as it felt more like a gentle tap than an alarm.

#### 2.5.5 Summary of Findings

| Methodology | Key Finding | System Implication |
|---|---|---|
| Literature Review | Dynamic signal-quality fusion outperforms static fusion (Wei et al., 2018) | Module C: Adaptive fusion weights |
| Literature Review | Multiresolution CNN handles sampling rate mismatch (John et al., 2021) | Module B: Dual-branch CNN-LSTM |
| Literature Review | Blank stare and head freezing indicate mind wandering (Bosch and D'Mello, 2021) | Module A: Blank stare index and head dynamics |
| Survey | 72% of students report frequent mind wandering | Validates system utility |
| Survey | Haptic vibration is the preferred intervention modality | Galaxy Watch vibration alert |
| Interview | Self-awareness, not surveillance, is the primary value proposition | Local-only processing, no cloud upload |
| Interview | Personalised baselines needed for individual variation | 3-minute HR/RMSSD calibration period |
| Prototyping | Alert overlay too aggressive; needs dismiss capability | "I'm Refocused" button with suppression timer |
| Prototyping | Users want visual confirmation of active sensing | Differentiated camera and watch status indicators |

### 2.6 Context Diagram

The context diagram positions the DipSEER system at the centre of four external entity interactions:

1. **Student User**: Provides biometric data (facial appearance, cardiac rhythm, wrist motion) and receives cognitive state feedback (dashboard visualisations, haptic alerts, desktop notifications).
2. **Samsung Galaxy Watch**: Streams real-time PPG heart rate and IMU rotation vector sensor data via WSS; receives haptic vibration trigger commands.
3. **Webcam Hardware**: Provides continuous video frame data to the visual processing pipeline.
4. **Web Browser**: Receives real-time telemetry broadcasts and renders the monitoring dashboard.

The system boundary encompasses all processing logic (visual inference, physiological inference, fusion, intervention dispatch) and excludes external hardware sensors and display devices.

### 2.7 Use Case Diagram

The use case diagram identifies seven primary use cases:

1. **UC1: Monitor Cognitive Focus** — The student user starts a monitoring session through the web dashboard.
2. **UC2: Detect Mind Wandering (Visual)** — Module A analyses webcam frames to compute visual engagement probability.
3. **UC3: Detect Mind Wandering (Physiological)** — Module B processes Galaxy Watch telemetry to compute physiological focus probability.
4. **UC4: Fuse Modality Scores** — Module C dynamically weights and integrates modality scores.
5. **UC5: Trigger Haptic Alert** — The system sends a vibration command to the Galaxy Watch upon detecting zoned-out state.
6. **UC6: Display Desktop Alert** — The system renders a dismissable overlay notification on the laptop.
7. **UC7: View Real-Time Dashboard** — The student views live engagement metrics, modality weights, and connection status.

### 2.8 Use Case Descriptions

**UC1: Monitor Cognitive Focus**

| Field | Description |
|---|---|
| Actor | Student User |
| Precondition | Laptop backend is running; webcam is functional |
| Main Flow | 1. User opens the web dashboard at http://localhost:5173. 2. System establishes WSS connection to backend. 3. System begins 1 Hz inference loop. 4. Dashboard displays real-time cognitive state updates. |
| Alternative Flow | If Galaxy Watch is not connected, system operates in visual-only mode with 100% visual weighting. |
| Postcondition | Student receives continuous cognitive state feedback until session ends. |

**UC5: Trigger Haptic Alert**

| Field | Description |
|---|---|
| Actor | System (automated) |
| Precondition | Galaxy Watch is connected and worn; state classified as "zoned out" |
| Main Flow | 1. Module C classifies fused score as below threshold (< 0.45). 2. System constructs JSON vibration command. 3. Command is broadcast to all connected watch clients. 4. DipSeerService receives command and activates vibration motor. 5. Double-pulse waveform (300ms–150ms–300ms) is executed. |
| Alternative Flow | If last vibration was within 4-second cooldown period, vibration is suppressed to prevent annoyance. |
| Postcondition | User receives wrist-based tactile feedback alerting them to refocus. |

### 2.9 Requirements

#### 2.9.1 Functional Requirements

| ID | Requirement | Priority | Linked UC |
|---|---|---|---|
| FR01 | The system shall capture webcam frames at a minimum of 20 FPS and perform real-time face detection using Haar cascade classifiers. | Must Have | UC2 |
| FR02 | The system shall decompose the ocular region into an 8×8 spatial energy grid and extract SVD-based engagement features. | Must Have | UC2 |
| FR03 | The system shall receive real-time heart rate and rotation vector data from the Samsung Galaxy Watch via secure WebSocket. | Must Have | UC3 |
| FR04 | The system shall compute HRV metrics (SDNN, RMSSD) from incoming heart rate data with personalised baseline calibration. | Must Have | UC3 |
| FR05 | The system shall fuse visual and physiological modality scores using dynamic signal-quality adaptive weighting. | Must Have | UC4 |
| FR06 | The system shall classify the user's cognitive state as "focused", "zoned out", or "user away" at 1 Hz frequency. | Must Have | UC1 |
| FR07 | The system shall trigger a haptic vibration on the Galaxy Watch when mind wandering is detected. | Should Have | UC5 |
| FR08 | The system shall display a dismissable desktop alert overlay when mind wandering is detected. | Should Have | UC6 |
| FR09 | The system shall display real-time engagement metrics, modality weights, and connection status on the web dashboard. | Should Have | UC7 |
| FR10 | The system shall automatically discover the laptop's IP address on the local network and configure smartwatch connectivity without manual input. | Should Have | UC3 |
| FR11 | The system shall display the watch wear status (disconnected, off-body, active) on the dashboard. | Could Have | UC7 |
| FR12 | The system shall play an audible system chime alongside the desktop notification when mind wandering is detected. | Could Have | UC6 |

#### 2.9.2 Non-Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| NFR01 | All biometric data processing shall occur locally on the user's laptop without cloud transmission. | Must Have |
| NFR02 | The inference loop shall execute within 1 second per cycle on a standard laptop CPU without GPU. | Must Have |
| NFR03 | WebSocket communication between all components shall be encrypted using TLS (WSS). | Must Have |
| NFR04 | The dashboard UI shall render at 60 FPS with smooth CSS transitions and animations. | Should Have |
| NFR05 | The system shall support graceful degradation when one or more modalities are unavailable. | Must Have |
| NFR06 | The Galaxy Watch application shall maintain foreground service with persistent notification during operation. | Must Have |

### 2.10 Chapter Summary

This chapter has documented the systematic requirement elicitation process for DipSEER through four complementary methodologies. The literature review established the scientific foundation for dynamic multimodal fusion and SVD-based visual analysis. The survey validated the prevalence of mind wandering among the target population and confirmed haptic vibration as the preferred intervention modality. Interviews emphasised self-awareness and privacy as core user values. Prototyping feedback informed specific UI design decisions including dismissable alerts and differentiated status indicators. The resulting requirements catalogue comprises 12 functional and 6 non-functional requirements spanning visual processing, physiological inference, dynamic fusion, intervention delivery, and data privacy.

---

## 3. DESIGN

### 3.1 Chapter Overview

This chapter presents the architectural design of the DipSEER system across multiple levels of abstraction, from the high-level three-tier architecture to low-level component interactions, data flow, and user interface design. Design decisions are justified with reference to the requirements established in Chapter 2 and the academic foundations identified in the literature review.

### 3.2 Design Goals

| Goal | Rationale | Linked Requirements |
|---|---|---|
| **Modular Pipeline Architecture** | Each sensing modality should operate independently, enabling graceful degradation when one modality is unavailable (NFR05). | FR05, NFR05 |
| **Sub-Second Inference Latency** | Real-time feedback is essential for timely re-engagement; delays exceeding 2 seconds render interventions psychologically ineffective. | NFR02 |
| **Zero-Configuration Networking** | Students should not need to configure IP addresses or firewall rules — the system must "just work" on any local network. | FR10 |
| **Privacy by Design** | No biometric data should leave the user's device; video frames should be processed in memory and never written to disk. | NFR01 |
| **Adaptive Fusion Resilience** | The fusion engine should dynamically redistribute modality weights in response to signal degradation rather than failing silently. | FR05 |
| **Multi-Device Intervention** | Interventions should reach the user through multiple channels simultaneously (visual, auditory, haptic) to maximise re-engagement probability. | FR07, FR08, FR12 |

### 3.3 High Level Design

#### 3.3.1 Architecture Diagram

The DipSEER system employs a three-tier edge computing architecture where all processing occurs locally on the user's devices:

**Tier 1 — Sensing Layer:** Comprises the laptop webcam (USB/integrated) capturing video at 20 FPS and the Samsung Galaxy Watch 7 streaming PPG heart rate and 3-axis rotation vector data via Wear OS sensors.

**Tier 2 — Processing Layer:** The Python asyncio backend running on the laptop, which hosts three concurrent modules: Module A (visual SVD inference running on a daemon thread), Module B (physiological multiresolution CNN-LSTM inference), and Module C (dynamic fusion meta-classifier and intervention engine). An encrypted WebSocket Secure (WSS) server on port 8765 serves as the central communication hub.

**Tier 3 — Presentation Layer:** The Vite-powered web dashboard rendered in the user's browser, connecting to the backend via WSS and rendering real-time telemetry through animated gauges, state orbs, and alert overlays.

#### 3.3.2 Discussion of Tiers

The decision to implement a local edge computing architecture rather than a cloud-based pipeline was driven by three factors. First, privacy requirements (NFR01) mandate that facial video and physiological telemetry remain on the user's device. Second, latency requirements (NFR02) preclude network round-trips to remote servers. Third, the system must function without internet connectivity, as students may use the system in locations with unreliable network access.

The separation of the Galaxy Watch from the processing tier is necessitated by the computational limitations of Wear OS devices, which lack the processing power to execute CNN-LSTM inference or OpenCV-based face detection. The watch therefore functions exclusively as a sensor streaming node and haptic actuator, delegating all inference to the laptop.

The communication infrastructure employs a hub-and-spoke topology with the Python backend as the central hub. The Galaxy Watch communicates via WSS (bidirectional: sensor data upstream, haptic commands downstream), while the web dashboard communicates via WSS (unidirectional: telemetry broadcast downstream). A UDP broadcast beacon on port 8766 enables zero-configuration discovery of the backend's IP address by the watch.

### 3.4 Low Level Design

#### 3.4.1 Choice of Design Paradigm

The system employs an **event-driven asynchronous** design paradigm, implemented using Python's `asyncio` framework. This choice is motivated by the concurrent nature of the system's operations: the inference loop, WebSocket server, and UDP discovery beacon must execute concurrently without blocking each other. The asyncio event loop manages I/O-bound operations (WebSocket send/receive, UDP broadcast) efficiently through cooperative multitasking, while the computationally intensive visual capture loop runs on a separate daemon thread to avoid blocking the event loop.

### 3.5 Design Diagrams

#### 3.5.1 Component Diagram

The component diagram identifies seven principal components and their interfaces:

1. **VisualInferenceModel** (Module A) — Exposes `predict()`, `get_image_traits()`, and `get_visual_quality()` interfaces. Internally manages the OpenCV VideoCapture, Haar cascade detector, SVD decomposition, TSFEL feature extraction, and XGBoost classifier.

2. **PhysioInferenceModel** (Module B) — Exposes `update_buffer(payload)`, `predict()`, `get_physio_quality()`, `is_sensor_worn()`, and `get_latest_metrics()` interfaces. Internally manages rolling deque buffers, HRV computation, baseline calibration, and TensorFlow Keras model inference.

3. **MetaClassifier** (Module C) — Exposes `classify(v_prob, p_prob, is_calibrating, watch_connected, q_vis, q_phys)` returning a 4-tuple of `(state, fused_score, w_vis, w_phys)`.

4. **AlertEngine** — Exposes `trigger_intervention(state)` with internal cooldown management and subprocess-based PowerShell invocation.

5. **DiscoveryBeacon** — Broadcasts UDP packets on all active subnet broadcast addresses using `psutil` network interface enumeration.

6. **SSLGenerator** — Generates self-signed X.509 TLS certificates with multi-IP Subject Alternative Name (SAN) entries.

7. **DipSeerService** (Wear OS) — Manages sensor event listeners, OkHttp WebSocket client, UDP discovery receiver, and haptic vibration actuator.

#### 3.5.2 Data Flow Diagram

The Level 1 data flow diagram traces the transformation of raw sensor inputs into classified cognitive states and intervention actions:

**Data Flow 1: Visual Pipeline.** Raw video frames (640×480, BGR) → Face detection (frontal/profile cascade) → Ocular region extraction (128×96, grayscale) → 8×8 block energy computation (64-D vector) → SVD decomposition (U, Σ, V^T) → FFT-based singular vector selection → TSFEL feature extraction (140+ statistical/spectral features) → XGBoost probability prediction → Calibrated visual engagement score.

**Data Flow 2: Physiological Pipeline.** Galaxy Watch JSON packets (HR bpm, rotation x/y/z/w) → JSON parsing and type routing → Heart rate → RR interval computation → SDNN/RMSSD calculation → Personalised baseline normalisation → 3-D physiological feature vector → 10-timestep sliding window → Low-resolution 1D-CNN (k=2, 16 filters) → Dual pooling (avg + max) → Concatenation with visual branch → LSTM (32 units) → Focus probability.

**Data Flow 3: Fusion and Intervention.** Visual probability + Visual quality + Physiological probability + Physiological quality → Dynamic weight computation (Wei et al., 2018) → Weighted fusion → Binary classification ("focused" / "zoned out" / "user away") → Parallel dispatch: (a) JSON telemetry → Dashboard broadcast, (b) Haptic command → Watch broadcast, (c) Desktop notification → PowerShell subprocess.

#### 3.5.3 System Process Activity Diagram

The activity diagram models the 1 Hz inference cycle as a sequential pipeline with parallel fan-out at the intervention stage:

1. START → Capture latest visual probability and 4D image traits
2. → Compute visual quality Q_vis
3. → Compute physiological probability from CNN-LSTM
4. → Compute physiological quality Q_phys
5. → Check watch connectivity and wear status
6. → DECISION: Face detected?
   - NO → Set state to "user away", skip fusion
   - YES → Continue to fusion
7. → Compute dynamic fusion weights (W_vis, W_phys)
8. → Calculate fused score
9. → DECISION: Fused score < 0.45?
   - YES → State = "zoned out" → PARALLEL: {Trigger desktop alert, Send watch haptic, Broadcast telemetry}
   - NO → State = "focused" → Broadcast telemetry
10. → Sleep 1 second → LOOP to Step 1

#### 3.5.4 User Interface Design

The web dashboard employs a cyberpunk-inspired glassmorphism design language with a dark navy background (`#0f172a`), semi-transparent glass cards with `backdrop-filter: blur(16px)`, and accent colours that semantically encode cognitive state: cyan (`#22d3ee`) for focused/active indicators, red (`#ef4444`) for zoned-out/alert states, and amber (`#f59e0b`) for warning/standby conditions.

The layout comprises a responsive two-column CSS Grid containing four cards:

1. **Module A Card (Visual Processing):** Displays a camera status banner ("Tracking Face (Active)" / "No Face Detected"), a progress bar visualising visual engagement probability, and a dynamic weight badge showing the current visual fusion weight percentage.

2. **Module B Card (Physiological Data):** Contains a watch status banner ("Watch Disconnected" / "Watch Connected (Off-Body / Acquiring Pulse)" / "Tracking Physiological Telemetry (Active)"), heart rate and HRV readouts, a progress bar for physiological focus level, and a dynamic weight badge.

3. **Module C Card (Meta-Classifier Decision):** Features a large glowing orb that pulses green when focused, red when zoned out, and grey when user is away. Below the orb, the current state label is displayed in uppercase text.

4. **Alert Overlay:** A fullscreen semi-transparent red overlay with a centred modal containing an animated alarm emoji, "ATTENTION REQUIRED" heading, explanatory text, and an "I'm Refocused 🎯" dismiss button. The overlay can also be dismissed with Space or Escape keys. After dismissal, alerts are suppressed for 8 seconds to prevent re-triggering during the recovery period.

### 3.6 Chapter Summary

This chapter has presented the DipSEER system design across high-level three-tier architecture, low-level event-driven asynchronous component interactions, data flow transformations, and user interface layout. The design prioritises modular independence of sensing modalities, sub-second inference latency, zero-configuration networking, and privacy-preserving local processing. The following chapter describes the initial implementation of these design specifications.

---

## 4. INITIAL IMPLEMENTATION

### 4.1 Chapter Overview

This chapter describes the technology selection rationale, the implementation of core functionality across all three modules, and the integration of the complete pipeline from sensor capture through inference, fusion, and intervention delivery.

### 4.2 Technology Selection

#### 4.2.1 Technology Stack

The technology stack was selected to satisfy the constraints identified in Chapter 2 and Chapter 3: local CPU-only inference, real-time performance, cross-platform wearable communication, and privacy-preserving data processing.

#### 4.2.2 Dataset Selection

Two complementary datasets were selected to train the visual and physiological models:

**DAiSEE Dataset (Dataset for Affective States in E-Environments).** This publicly available benchmark dataset comprises 9,068 ten-second video clips recorded from 112 subjects in simulated e-learning scenarios. Each clip is annotated with four affective states — Boredom, Engagement, Confusion, and Frustration — rated on a 4-level scale (0 = Very Low, 1 = Low, 2 = High, 3 = Very High). For this project, the Engagement dimension was binarised: levels 2 and 3 (High/Very High engagement) were mapped to "Focused" (label 1), while levels 0 and 1 (Very Low/Low engagement) were mapped to "Zoned Out" (label 0). The dataset was used exclusively for training Module A's SVD-TSFEL-XGBoost visual classifier.

**DipSeer Dataset.** This multimodal dataset was collected from 21 subjects across 65 recording sessions, capturing synchronised Samsung Galaxy Watch sensor streams (PPG heart rate, 3-axis rotation vector, linear acceleration, gyroscope, ambient light) alongside self-reported attention labels on a 5-level scale at regular intervals. The attention labels were mapped to continuous focus probabilities: Level 1 → 0.15 (Zoned Out), Level 2 → 0.40 (Mind Wandering), Level 3 → 0.72 (Normal Focus), Level 4 → 0.86 (High Focus), Level 5 → 0.95 (Deep Focus). This dataset was used to train Module B's Multiresolution CNN-LSTM model.

#### 4.2.3 Programming Languages

**Python 3.10** was selected as the primary backend language due to its extensive ecosystem of scientific computing libraries (NumPy, TensorFlow, OpenCV, TSFEL), native asyncio support for concurrent I/O operations, and broad community support for machine learning development.

**Kotlin** was selected for the Wear OS Galaxy Watch application, as it is the official first-class language for Android and Wear OS development, offering null safety, coroutine support for asynchronous sensor operations, and interoperability with the Android Sensor Framework and Jetpack Compose UI toolkit.

**JavaScript (ES2022)** was used for the web dashboard frontend, leveraging modern browser APIs for WebSocket communication and CSS animation.

#### 4.2.4 Development Frameworks

| Framework | Purpose | Justification |
|---|---|---|
| TensorFlow 2.x with Keras | CNN-LSTM model training and CPU inference | Mature model serialization (H5), CPU-only execution support via device pinning |
| OpenCV 4.x | Webcam capture, face detection, image processing | Industry-standard real-time computer vision library with DirectShow backend for Windows |
| Vite 8.x | Frontend web dashboard build tooling | Instant hot module replacement, minimal configuration, fast development iterations |
| OkHttp3 | Wear OS WebSocket client | Robust HTTP/WebSocket library for Android with built-in reconnection support |
| Jetpack Compose for Wear OS | Watch application UI | Declarative UI toolkit native to Wear OS, reducing boilerplate code |

#### 4.2.5 Libraries

| Library | Module | Purpose |
|---|---|---|
| NumPy | A, B | Matrix operations, SVD decomposition, statistical computations |
| TSFEL | A | Automated extraction of 140+ temporal, statistical, and spectral features from SVD singular vectors |
| XGBoost | A | Gradient-boosted tree classifier for visual engagement probability estimation |
| scikit-learn | Training | Data splitting, metric evaluation, preprocessing |
| pandas | Training | Temporal merging of sensor streams with attention labels using `merge_asof` |
| websockets | Backend | Asyncio-native WebSocket server implementation for Python |
| psutil | Backend | Cross-platform network interface enumeration for multi-subnet UDP discovery |
| cryptography | Backend | X.509 certificate generation for TLS-encrypted WebSocket communication |
| winsound | Backend | Native Windows system audio for alert beep generation |

#### 4.2.6 IDEs

**Visual Studio Code** was used for Python backend development and web dashboard development, with the Python, Pylint, and Live Server extensions. **Android Studio (Hedgehog)** was used for Kotlin Wear OS application development, providing the Wear OS emulator, ADB debugging, and Gradle build management.

#### 4.2.7 Summary of Technology Selection

| Layer | Technology | Language | Runtime |
|---|---|---|---|
| Visual Inference (Module A) | OpenCV, NumPy SVD, TSFEL, XGBoost | Python 3.10 | Daemon thread |
| Physio Inference (Module B) | TensorFlow Keras CNN-LSTM (CPU) | Python 3.10 | Asyncio event loop |
| Fusion Engine (Module C) | NumPy weighted fusion | Python 3.10 | Asyncio event loop |
| WebSocket Server | websockets, ssl | Python 3.10 | Asyncio event loop |
| Discovery Beacon | socket, psutil | Python 3.10 | Asyncio event loop |
| Wearable App | Android Sensor Framework, OkHttp3 | Kotlin | Wear OS Foreground Service |
| Web Dashboard | Vanilla JS, Vite, CSS3 | JavaScript | Browser |

### 4.3 Implementation of Core Functionality

**Module A: SVD-Based Visual Inference Pipeline**

The visual processing pipeline implements a novel multi-stage feature extraction approach. The OpenCV VideoCapture thread runs continuously at 20 FPS, applying a dual-cascade face detection strategy: the frontal face cascade (`haarcascade_frontalface_alt2.xml`) is attempted first, and upon failure, the profile face cascade (`haarcascade_profileface.xml`) is applied to both the original frame and its horizontal mirror, enabling robust detection across frontal, three-quarter, and profile poses.

Upon successful face detection, the ocular region is extracted by cropping the upper 40% of the detected face bounding box (y ∈ [0.15h, 0.55h], x ∈ [0.1w, 0.9w]), converted to grayscale, and resized to 128×96 pixels. This fixed-size region is partitioned into 64 non-overlapping blocks of 16×12 pixels each, and the L₂ energy of each block is computed as the sum of squared pixel intensities. The resulting 64-dimensional energy vector is appended to a rolling buffer of 200 frames (representing approximately 10 seconds of data).

When sufficient frames have accumulated, the energy buffer matrix E (200×64) is decomposed using NumPy's SVD implementation: E = UΣV^T. The first 10 left singular vectors from U are each transformed via FFT (`np.fft.rfft`), and the vector exhibiting the highest high-frequency spectral amplitude is selected as the dominant eye movement pattern. The TSFEL library then extracts over 140 statistical, temporal, and spectral features from this selected singular vector at a sampling rate of 20 Hz. These features are input to a pre-trained XGBoost classifier, which outputs a raw focus probability. The probability is calibrated using power exponentiation (P_cal = P_raw^8.0) to sharpen the decision boundary between focused and unfocused states.

Building upon the foundational SVD pipeline, two additional behavioural indicators are computed following Bosch and D'Mello's (2021) framework. The **Blank Stare Fixation Index** analyses the temporal variance of block energies over the most recent 60 frames (3 seconds). If the micro-saccadic variation across all 64 ocular blocks falls below a threshold of 5.0, the system identifies a prolonged gaze fixation characteristic of blank staring during mind wandering episodes, assigning a low attention score of 0.30. The **Head Dynamics Stability** metric tracks the centroid of the detected face bounding box across a 10-second rolling window, computing the first-derivative velocity and spatial jitter. Active but controlled head movements (3–15 pixels/second) indicate engaged cognitive processing, while rigid freezing (<0.3 pixels/second) or fatigued drooping (>40 pixels/second) signal potential disengagement.

**Module B: Multiresolution Dual-Pooling CNN-LSTM Physiological Inference**

The physiological inference module addresses the fundamental challenge of temporal resolution mismatch between the 20 FPS visual pipeline and the approximately 1 Hz PPG heart rate stream from the Galaxy Watch. Following John et al.'s (2021) multiresolution data fusion architecture, the module employs dedicated 1D convolutional branches for each temporal resolution before cross-resolution concatenation.

The low-resolution physiological branch processes a 10-timestep sliding window of 3-dimensional normalised features: ΔHR (deviation from personalised median baseline), ΔRMSSD (deviation from calibrated variability baseline), and normalised SDNN. Each feature is computed relative to a personalised rolling baseline maintained over 180 readings (approximately 3 minutes of continuous monitoring), enabling the system to adapt to individual cardiac profiles.

The high-resolution visual/motion branch processes a concurrent 10-timestep window of 4-dimensional image traits extracted from Module A: gaze energy (SVD-derived eye movement intensity), blank stare metric, head dynamics stability, and calibrated visual engagement probability.

Each branch passes through a 1D convolutional layer (physio: 16 filters, kernel size 2; visual: 32 filters, kernel size 3), followed by batch normalisation and **dual temporal pooling** inspired by Najafi et al. (2023). This dual pooling applies both AveragePooling1D and MaxPooling1D in parallel, concatenating their outputs along the channel dimension. Average pooling captures sustained autonomic trends (gradual heart rate elevation during fatigue), while maximum pooling preserves transient cardiac spikes and abrupt gaze shifts that may indicate momentary re-engagement or startle responses.

The concatenated dual-branch features are fed into a 32-unit LSTM layer that models temporal dependencies across the fused multiresolution representation, followed by batch normalisation, dropout (0.2), a 16-unit dense layer with ReLU activation, and a single sigmoid output neuron predicting focus probability in the continuous range [0.0, 1.0].

**Module C: Dynamic Signal-Quality Adaptive Fusion Engine**

The meta-classifier implements Wei et al.'s (2018) signal-quality adaptive weighted fusion strategy, extending the original emotion recognition application to cognitive focus detection. Rather than employing fixed fusion weights (e.g., a static 65% physiological / 35% visual split), the system continuously estimates the real-time quality of each modality and adjusts weights accordingly.

The **visual quality index** Q_vis ∈ [0.2, 1.0] is computed from two factors: face bounding box resolution (normalised against an optimal width of 160 pixels on a 640×480 frame) and head tracking stability (from the head dynamics metric). When the user leans close to the camera, Q_vis approaches 1.0; when they turn away or move to the edge of the frame, Q_vis decreases, reducing the visual modality's influence.

The **physiological quality index** Q_phys ∈ [0.35, 1.0] is derived from the Galaxy Watch's IMU rotation vector. During calm desk work, the wrist remains relatively stationary and Q_phys ≈ 0.95. During typing or gesticulating, increased wrist angular velocity introduces PPG motion artefacts, and Q_phys decreases proportionally. If no heart rate reading is received for more than 5 seconds (indicating potential watch removal or sensor failure), Q_phys drops to 0.20.

The dynamic fusion weights are computed as:

$$W_{\text{phys}} = \frac{0.65 \cdot \max(0.1, Q_{\text{phys}})}{0.65 \cdot \max(0.1, Q_{\text{phys}}) + 0.35 \cdot \max(0.1, Q_{\text{vis}}) + \epsilon}$$

$$W_{\text{vis}} = 1.0 - W_{\text{phys}}$$

The base coefficients (0.65 for physiological, 0.35 for visual) encode the domain-informed prior that autonomic nervous system signals provide more reliable indicators of cognitive state than facial appearance alone, while the quality-modulated scaling ensures that degraded modalities are automatically downweighted.

The fused focus score is then computed as:

$$S_{\text{fused}} = W_{\text{vis}} \cdot P_{\text{vis}} + W_{\text{phys}} \cdot P_{\text{phys}}$$

Classification applies a threshold of 0.45: if S_fused < 0.45, the state is classified as "zoned out"; otherwise, it is "focused". When no face is detected, the state is set to "user away" regardless of physiological signals. The system operates across three distinct modes: full bimodal fusion when both modalities are available, visual-only fallback when the watch is disconnected, and a 90/10 visual-predominant mode during the initial 30-second physiological calibration period.

### 4.4 Chapter Summary

This chapter has detailed the implementation of DipSEER's three-module pipeline: the SVD-based visual inference engine with Bosch and D'Mello's behavioural indicators, the multiresolution dual-pooling CNN-LSTM physiological model, and the dynamic signal-quality adaptive fusion meta-classifier. The technology stack — comprising Python 3.10, TensorFlow Keras, OpenCV, XGBoost, TSFEL, Kotlin Wear OS, and a Vite web dashboard — was selected to satisfy the requirements of local CPU-only real-time inference, privacy-preserving processing, and zero-configuration smartwatch connectivity. The following chapter presents the evaluation of this initial implementation, discusses deviations from the original plan, and identifies areas requiring further development.

---

## 5. CONCLUSION

### 5.1 Chapter Overview

This chapter evaluates the current state of the DipSEER prototype, documents deviations from the original project plan, presents initial test results, identifies required improvements for the final submission, and summarises the demonstration of the working prototype.

### 5.2 Deviations

#### 5.2.1 Scope Related Deviations

The original project proposal included a Bluetooth Low Energy (BLE) GATT communication protocol between the Galaxy Watch and the laptop backend. During implementation, this was replaced with a WebSocket Secure (WSS) communication architecture for two reasons. First, BLE has a maximum theoretical throughput of approximately 1 Mbps with practical throughput significantly lower, which introduces latency in high-frequency sensor streaming. Second, BLE pairing on Windows is unreliable for sustained data transfer, with frequent connection drops observed during early prototyping. The WSS approach leverages the watch's existing Wi-Fi connection, providing higher throughput, lower latency, and more robust connection management through OkHttp's built-in reconnection mechanisms.

An additional scope expansion involved the implementation of the dynamic signal-quality adaptive fusion engine (Module C), which was not part of the original proposal but was added after literature review identified Wei et al.'s (2018) methodology as a significant improvement over the originally planned static fusion weights.

#### 5.2.2 Schedule Related Deviations

The DipSeer Dataset collection phase required two additional weeks beyond the original schedule due to sensor synchronisation challenges between the Galaxy Watch's internal clock and the laptop's system clock. This was resolved by implementing timestamp-tolerant temporal merging using pandas' `merge_asof` function with a 45-second tolerance window.

The transition from GPU to CPU inference for the TensorFlow models was necessitated by an incompatibility between DirectML TensorFlow on Windows and standard Keras LSTM operations, which throw `No OpKernel was registered to support Op 'CudnnRNN'` errors. All model training and inference were reconfigured to execute on CPU using `tf.config.set_visible_devices([], 'GPU')`, achieving forward pass latencies below 3 milliseconds despite the absence of GPU acceleration.

### 5.3 Initial Test Results

The dynamic signal-quality adaptive fusion engine was verified across four representative operational scenarios:

| Scenario | Q_vis | Q_phys | Dynamic Split | Observed Behaviour |
|---|---|---|---|---|
| Normal Desk Study | 1.00 | 0.95 | 64% Physio / 36% Visual | Full bimodal fusion with physiological priority. System accurately classifies focused state during reading. |
| Typing / Wrist Movement | 1.00 | 0.35 | 39% Physio / 61% Visual | System shifts weight to camera, bypassing PPG motion noise. Prevents false zoned-out classifications during note-taking. |
| Head Turn / Dim Lighting | 0.30 | 1.00 | 86% Physio / 14% Visual | System relies primarily on Galaxy Watch when camera loses tracking. Maintains classification accuracy during temporary visual occlusion. |
| Watch Disconnected | 1.00 | 0.00 | 0% Physio / 100% Visual | Clean visual-only fallback without system crash or error. Dashboard correctly shows "Watch Disconnected" status. |

The Multiresolution CNN-LSTM model trained on the DipSeer Dataset (5,434 sequences, 21 subjects) achieved a validation loss of 0.000065 (MSE) and a mean absolute error (MAE) of 0.0062 on the held-out validation set. While these metrics are promising, further evaluation on unseen subjects (leave-one-subject-out cross-validation) is required to assess generalisation performance.

### 5.4 Required Improvements

The following improvements are identified for the final submission:

1. **Leave-one-subject-out cross-validation** for the multiresolution CNN-LSTM model to assess inter-subject generalisation.
2. **Extended calibration persistence** to save personalised baselines across sessions rather than recalibrating from scratch on each startup.
3. **Session logging and analytics** to provide post-study summaries of attentional patterns, including mind wandering frequency, average focus duration, and temporal distribution of attention lapses.
4. **Attention decay modelling** to detect gradual disengagement trends over extended study sessions, rather than binary focused/zoned-out classification.
5. **User study evaluation** with a sample of 10–15 participants comparing mind wandering detection accuracy against ground truth experience sampling probes.

### 5.5 Demo of the Prototype

The prototype was demonstrated under live conditions with the following configuration: a Windows 11 laptop with Intel Core i5 processor (no dedicated GPU), integrated webcam, Samsung Galaxy Watch 7 (Wear OS 5.0) connected via Wi-Fi, and Google Chrome rendering the web dashboard.

The demonstration showcased:
- Real-time face detection and SVD-based visual engagement estimation with the "Tracking Face (Active)" indicator.
- Live Galaxy Watch connectivity with automatic IP discovery, heart rate streaming, and differentiated watch status transitions from "Watch Disconnected" → "Watch Connected (Off-Body / Acquiring Pulse)" → "Tracking Physiological Telemetry (Active)".
- Dynamic fusion weight badges updating in real-time on the dashboard (e.g., "Weight: 64%" for physiological, "Weight: 36%" for visual during normal desk study).
- Successful mind wandering detection triggering simultaneous Galaxy Watch haptic vibration (double-pulse waveform), Windows desktop balloon notification with audio chime, and interactive full-screen dashboard alert overlay with dismiss functionality.
- Graceful degradation when the watch was removed from the wrist, with the system smoothly transitioning to visual-only mode.

### 5.6 Chapter Summary

This chapter has documented the current state of the DipSEER prototype, including scope expansions (WSS communication, dynamic adaptive fusion), schedule adjustments (CPU-only inference, dataset synchronisation), and initial verification results demonstrating correct dynamic weight adaptation across four operational scenarios. The prototype successfully integrates SVD-based visual analysis, multiresolution CNN-LSTM physiological inference, and dynamic signal-quality adaptive fusion with multi-device closed-loop interventions. Required improvements for the final submission include cross-validation evaluation, session persistence, analytics logging, and a formal user study.

---

## REFERENCES

Baldwin, C.L., Roberts, D.M., Barragan, D., Lee, J.D., Lerner, N. and Higgins, J.S. (2017) 'Detecting and quantifying mind wandering during simulated driving', *Frontiers in Human Neuroscience*, 11, p. 406.

Bhatia, S., Mitra, S. and Gupta, G. (2021) 'SVD-Based Mind-Wandering Prediction from Facial Videos in Online Learning', *IEEE Access*, 9, pp. 150689–150700.

Bixler, R. and D'Mello, S. (2016) 'Automatic gaze-based user-independent detection of mind wandering during computerized reading', *User Modeling and User-Adapted Interaction*, 26(1), pp. 33–68.

Bosch, N. and D'Mello, S. (2021) 'Automatic Detection of Mind Wandering from Video in the Lab and in the Classroom', *IEEE Transactions on Affective Computing*, 12(4), pp. 974–988.

Braun, V. and Clarke, V. (2006) 'Using thematic analysis in psychology', *Qualitative Research in Psychology*, 3(2), pp. 77–101.

Can, Y.S., Chalabianloo, N. and Ersoy, C. (2019) 'A Review on Mental Stress Detection Using Wearable Sensors and Machine Learning Techniques', *IEEE Access*, 7, pp. 131512–131531.

D'Mello, S. (2016) 'Giving eyesight to the blind: towards attention-aware AIED', *International Journal of Artificial Intelligence in Education*, 26(2), pp. 645–659.

D'Mello, S., Mills, C. and Bixler, R. (2021) 'Mind Wandering in a Multimodal Reading Setting: Behavior Analysis and Automatic Detection Using Eye-Tracking and an EDA Sensor', *Journal of Educational Psychology*, 113(5), pp. 1029–1048.

Dong, H., Hu, B. and Liu, S. (2021) 'Mind wandering state detection during video-based learning via EEG', *Biomedical Signal Processing and Control*, 68, p. 102742.

Faber, M., Bixler, R. and D'Mello, S.K. (2018) 'An automated behavioral measure of mind wandering during computerized reading', *Behavior Research Methods*, 50(1), pp. 134–150.

Giannakos, M.N., Sharma, K., Pappas, I.O., Kostakos, V. and Velloso, E. (2022) 'Fusing Wearable Biosensors with Artificial Intelligence for Mental Health Monitoring: A Systematic Review', *Information Fusion*, 87, pp. 132–151.

Gjoreski, M., Gjoreski, H., Luštrek, M. and Gams, M. (2020) 'Cognitive Load Monitoring With Wearables: Lessons Learned From a Machine Learning Challenge', *IEEE Access*, 8, pp. 103670–103685.

Grandchamp, R., Braboszcz, C. and Delorme, A. (2014) 'Electrophysiological markers of mind wandering: A systematic review', *NeuroImage*, 84, pp. 489–502.

Hickey, B.A., Chalmers, T., Newton, P., Lin, C.T., Sibbritt, D., McLachlan, C.S., Clifton-Bligh, R., Morley, J. and Lal, S. (2021) 'Smart Devices and Wearable Technologies to Detect and Monitor Mental Health Conditions and Stress: A Systematic Review', *Sensors*, 21(10), p. 3461.

Hutt, S., Mills, C., Bosch, N., Krasich, K., Brockmole, J.R. and D'Mello, S. (2017) 'When Eyes Wander Around: Mind-Wandering as Revealed by Eye Movement Analysis with Hidden Markov Models', *Proceedings of the 39th Annual Conference of the Cognitive Science Society*, pp. 2285–2290.

John, A., Nundy, K.K., Cardiff, B. and John, D. (2021) 'Multimodal Multiresolution Data Fusion Using Convolutional Neural Networks for IoT Wearable Sensing', *IEEE Transactions on Biomedical Circuits and Systems*, 15(6), pp. 1161–1173.

Killingsworth, M.A. and Gilbert, D.T. (2010) 'A wandering mind is an unhappy mind', *Science*, 330(6006), p. 932.

Lim, S., Lee, G. and Kim, S. (2020) 'Predicting Mind-Wandering with Facial Videos in Online Lectures', *Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization*, pp. 360–364.

Najafi, T., Jafarzadeh, M., Khadem, A. and Ghaffari, M. (2023) 'Multi-Scale 1D-CNN for physiological time-series classification with dual temporal pooling', *Sensors*, 23(7), p. 3502.

Pielot, M., Church, K. and de Oliveira, R. (2014) 'An In-Situ Study of Mobile Phone Notifications', *Proceedings of the 16th International Conference on Human-Computer Interaction with Mobile Devices and Services*, pp. 233–242.

Risko, E.F., Anderson, N., Sarwal, A., Engelhardt, M. and Kingstone, A. (2012) 'Everyday attention: Variation in mind wandering and memory in a lecture', *Applied Cognitive Psychology*, 26(2), pp. 234–242.

Sharma, K., Castellini, C., van den Broek, E.L.,"; Albu-Schaffer, A. and Schwenker, F. (2019) 'A Survey on Wearable Sensors for Mental Health Monitoring', *Sensors*, 19(5), p. 1113.

Smallwood, J. and Schooler, J.W. (2015) 'The science of mind wandering: empirically navigating the stream of consciousness', *Annual Review of Psychology*, 66, pp. 487–518.

Smallwood, J., Davies, J.B., Heim, D., Finnigan, F., Sudberry, M., O'Connor, R. and Obonsawin, M. (2004) 'Subjective experience and the attentional lapse: Task engagement and disengagement during sustained attention', *Consciousness and Cognition*, 13(4), pp. 657–690.

Sommerville, I. (2015) *Software Engineering*. 10th edn. Harlow: Pearson Education.

Stewart, A., Cardiel, C. and Guo, A. (2020) 'Assessing student engagement from facial behavior in online learning', *Educational Technology Research and Development*, 68(4), pp. 1677–1695.

Wammes, J.D., Boucher, P.O. and Seli, P. (2016) 'Mind wandering during lectures I: Changes in rates across an entire semester', *Scholarship of Teaching and Learning in Psychology*, 2(1), pp. 33–48.

Wei, W., Jia, Q., Feng, Y. and Chen, G. (2018) 'Emotion Recognition Based on Weighted Fusion Strategy of Multichannel Physiological Signals', *Computational Intelligence and Neuroscience*, 2018, Article ID 5294528.

Yin, Y., Zuo, Z., Li, P. and Wei, J. (2019) 'Fusion of Video and Inertial Sensing for Deep Learning Based Human Action Recognition', *Sensors*, 19(17), p. 3687.

Zhang, X., Yao, L., Wang, X., Monaghan, J.J., Mcalpine, D. and Zhang, Y. (2021) 'A Survey on Vision-Based Driver Distraction Analysis', *IEEE Transactions on Intelligent Transportation Systems*, 22(12), pp. 7836–7856.

Zhu, Z., Li, Y. and Chen, F. (2020) 'Drivers' Mental Engagement Analysis Using Multi-Sensor Fusion Approaches Based on Deep Convolutional Neural Networks', *IEEE Access*, 8, pp. 87083–87095.

Kim, J., Jang, S., Choi, S. and Kim, H. (2021) 'Multimodal Engagement Recognition from Image Traits Using Deep Learning Techniques', *Applied Sciences*, 11(15), p. 7139.

Tokuno, A., Kawai, R. and Kanazawa, H. (2023) 'The Emergence of AI-Based Wearable Sensors for Digital Health Technology: A Review', *Sensors*, 23(12), p. 5587.

Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X., Ben-Zeev, D. and Campbell, A.T. (2020) 'Wearable, Environmental, and Smartphone-Based Passive Sensing for Mental Health Monitoring', *IEEE Pervasive Computing*, 19(1), pp. 32–43.

Hutt, S., Krasich, K., Mills, C., Brockmole, J. and D'Mello, S. (2019) 'Webcam-Based Eye Tracking to Detect Mind Wandering and Comprehension Errors', *Behavior Research Methods*, 51(3), pp. 1050–1063.

Faber, M., Krasich, K. and D'Mello, S.K. (2020) 'Predicting cognitive scores from wearable-based digital physiological features using machine learning: data from a clinical trial in mild cognitive impairment', *BMC Medical Informatics and Decision Making*, 20(1), p. 247.

Ko, L.W., Komarov, O., Lin, C.T. and Jung, T.P. (2020) 'EEG Complexity Measures for Detecting Mind Wandering During Video-Based Learning', *Frontiers in Neuroscience*, 14, p. 569.

Kang, Y., Kim, J. and Park, S. (2022) 'Individual Student Attention Detection in Face-to-Face Classrooms Using Multimodal Facial and Wearable Data', *Sensors*, 22(6), p. 2354.

Li, C., Chen, Y. and Wu, J. (2021) 'Classification of Internal and External Distractions in an Educational VR Environment Using Multimodal Features', *IEEE Access*, 9, pp. 112753–112765.

Tobaldini, E., Nobili, L., Strada, S., Casali, K.R., Braghiroli, A. and Montano, N. (2020) 'An Effective Entropy Assisted Mind Wandering Detection System Using EEG Signals', *Entropy*, 22(5), p. 567.

Park, S., Kim, H. and Lee, J. (2021) 'Automatic Cognitive Fatigue Detection Using Wearable fNIRS and Machine Learning', *Sensors*, 21(18), p. 6234.

Zhang, L., Li, M. and Wang, H. (2023) 'Vision-Language Models can Identify Distracted Driver Behaviour from Naturalistic Videos', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pp. 2110–2119.

Lee, S., Park, J. and Kim, D. (2022) 'Fusion of Multi-Sensor-Based Biomechanical Gait Analysis Using Vision and Wearable Sensor', *Sensors*, 22(4), p. 1520.

Cheng, Y., Liu, W. and Li, R. (2023) 'High-Fidelity Eye, Head, Body, and World Tracking with a Wearable Device', *IEEE Transactions on Visualization and Computer Graphics*, 29(5), pp. 2592–2602.

Kwon, H., Kim, J. and Lee, S. (2022) 'Human Daily Indoor Action (HDIA) Dataset: Privacy-Preserving Human Action Recognition Using Infrared Camera and Wearable Armband Sensors', *Sensors*, 22(9), p. 3460.

Kim, M., Park, S. and Lee, J. (2023) 'Synergy Through Integration of Digital Cognitive Tests and Wearable Devices for Mild Cognitive Impairment Screening', *Digital Health*, 9, pp. 1–14.

---

## APPENDIX A – Thematic Analysis (Requirement Elicitation)

### A.1 Coding Scheme

Thematic analysis of interview transcripts followed Braun and Clarke's (2006) six-phase methodology: familiarisation, initial coding, theme generation, theme review, theme definition, and report production. Initial codes were generated inductively from the data, then iteratively refined into three overarching themes.

### A.2 Theme 1: Self-Awareness and Metacognitive Support

Codes: *self-monitoring*, *attention patterns*, *self-regulation*, *personal insight*, *metacognition*

All interviewees expressed the view that mind wandering detection should primarily serve as a self-awareness tool rather than an external surveillance mechanism. The educational psychologist emphasised that metacognitive awareness of attentional patterns is itself a key skill: knowing *when* one tends to lose focus enables proactive strategy adjustment (e.g., scheduling breaks before predicted attention troughs).

### A.3 Theme 2: Privacy, Data Sovereignty, and Trust

Codes: *data locality*, *no cloud upload*, *camera recording anxiety*, *biometric sensitivity*, *transparency*

Privacy emerged as the strongest prerequisite across all interviews. The student representative articulated a clear boundary between acceptable local processing and unacceptable remote data transmission. This theme directly informed NFR01 (local-only processing) and the system's design decision to process video frames in volatile memory without disk persistence.

### A.4 Theme 3: Adaptive Personalisation

Codes: *individual variation*, *baseline differences*, *fidgeting vs. disengagement*, *learning curve*, *sensitivity tuning*

The lecturer's observation that baseline activity levels vary significantly across individuals motivated the personalised calibration mechanism in Module B, where 3 minutes of initial heart rate data establish an individual's resting cardiac profile before deviation-based mind wandering detection begins.

---

## APPENDIX B – UI Wireframes

### B.1 Web Dashboard Layout

The web dashboard wireframe depicts a two-column grid layout with four primary cards:

- **Top-Left:** Module A (Visual Processing) — Camera status indicator, visual engagement progress bar, dynamic weight badge.
- **Top-Right:** Module B (Physiological Data) — Watch status indicator, HR/HRV readouts, physio focus progress bar, dynamic weight badge.
- **Bottom (Full-Width):** Module C (Meta-Classifier Decision) — Central glowing state orb, classification label.
- **Overlay:** Alert modal with emoji, heading, subtext, and dismiss button.

### B.2 Galaxy Watch Interface

The Galaxy Watch wireframe shows a minimalist circular Wear OS interface with:

- Application title ("DipSEER Wear")
- Connection status text
- Start/Stop toggle button styled with Wear OS Material Design components.

---

## APPENDIX C – Component Diagram

The component diagram illustrates the seven principal software components and their interfaces as described in Section 3.5.1. The diagram uses UML component notation with «component» stereotypes and interface lollipop/socket connectors. Key dependency relationships include:

- `main.py` depends on `VisualInferenceModel`, `PhysioInferenceModel`, `MetaClassifier`, `AlertEngine`, and `DiscoveryBeacon`.
- `PhysioInferenceModel` depends on `VisualInferenceModel` (receives 4D image traits via `update_image_traits()`).
- `MetaClassifier` depends on quality metrics from both `VisualInferenceModel` (`get_visual_quality()`) and `PhysioInferenceModel` (`get_physio_quality()`).
- `DiscoveryBeacon` depends on `SSLGenerator` for certificate presence validation.
- `DipSeerService` (Wear OS) communicates with `main.py` (WSS server) bidirectionally.
