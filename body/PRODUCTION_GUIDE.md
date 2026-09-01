# Karma Robot — Production & 3D Printing Manufacturing Guide

An end-to-end industrial manufacturing, 3D printing, hardware assembly, and Bill of Materials (BOM) manual for the **Karma Autonomous Multimodal AI Companion Robot**.

---

## 1. Executive Summary & Robot Mechanical Specifications

| Parameter | Specification | Details / Notes |
| :--- | :--- | :--- |
| **Total Height** | **940 mm** (~37.0 inches) | From base floor to top of head assembly |
| **Total Width** | **320 mm** (Base) / **300 mm** (Head) | Broad stability footprint |
| **Total Depth** | **400 mm** (Base) / **210 mm** (Column) | Weighted anti-tip geometry |
| **Total Weight (Assembled)** | **~4.8 kg – 5.5 kg** | Including compute, 12V battery, screen & hardware |
| **Neck Actuation (Pitch DOF)**| **90° to 135° Range of Motion** | 90° (horizontal eye-level) to 135° (looking 45° down at desk) |
| **Actuator** | **20kg–25kg Digital Metal-Gear Servo** | Standard form factor (40.5 × 20.2 × 40.0 mm) |
| **Pivot Bearings** | **Dual 608ZZ Ball Bearings** | 8 mm ID × 22 mm OD × 7 mm Width |
| **Display** | **7.0" to 10.1" IPS Touchscreen** | HDMI / DSI interface (e.g. Waveshare IPS capacitive) |
| **Perception Vision** | **120° Wide-Angle HD Camera** | Top bezel mount for YOLOv8 object & face tracking |
| **Acoustic Audio** | **Dual 4Ω 5W 40mm/50mm Stereo Speakers** | Sealed internal acoustic chambers with forward grilles |
| **Compute Compatibility** | **Raspberry Pi 5 / Jetson Orin / Mini PC** | Universal 58×49mm & 100×100mm mounting sled in base |

---

## 2. Complete Bill of Materials (BOM)

### A. 3D Printed Structural Components (9 Parts)

| Part Name | STL File | Recommended Material | Walls / Perimeters | Infill Density | Est. Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Front Chassis** | `body/base_front.stl` | PETG / ABS / PLA+ | 5 walls (2.0 mm) | 30% Gyroid | ~480 g |
| **Base Back Chassis** | `body/base_back.stl` | PETG / ABS / PLA+ | 5 walls (2.0 mm) | 30% Gyroid | ~420 g |
| **Column Front Torso** | `body/column_front.stl` | PETG / ABS / PLA+ | 4 walls (1.6 mm) | 25% Gyroid | ~560 g |
| **Column Back Spine** | `body/column_back.stl` | PETG / ABS / PLA+ | 4 walls (1.6 mm) | 25% Gyroid | ~540 g |
| **Neck Front Actuation** | `body/neck_front.stl` | PETG / ABS-GF / PLA-CF | 6 walls (2.4 mm) | 40% Gyroid | ~160 g |
| **Neck Back Enclosure** | `body/neck_back.stl` | PETG / ABS / PLA+ | 4 walls (1.6 mm) | 30% Gyroid | ~110 g |
| **Steering Horn Linkage** | `body/steering_arm.stl` | PETG / Nylon / PLA-CF | 8 walls (Solid) | 100% Rectilinear | ~25 g |
| **Head Display Front Bezel**| `body/head_window_half.stl`| PETG / ABS / PLA+ | 4 walls (1.6 mm) | 25% Gyroid | ~280 g |
| **Head Rear Cover** | `body/head_cover_half.stl` | PETG / ABS / PLA+ | 4 walls (1.6 mm) | 20% Gyroid | ~210 g |
| **TOTAL FILAMENT** | — | — | — | — | **~2.78 kg** |

---

### B. Standard Fasteners & Mechanical Hardware

| Item | Specification | Qty | Purpose |
| :--- | :--- | :--- | :--- |
| **M4 Heat-Set Brass Inserts** | M4 × 6.0 mm OD × 8.0 mm Length | 24 | Base seam, Column flanges, Neck attachment |
| **M3 Heat-Set Brass Inserts** | M3 × 4.2 mm OD × 5.5 mm Length | 28 | Head bezel, Camera PCB, Compute sled, Servo mount |
| **M4 × 16 mm Socket Head Screws**| M4 × 16 mm Stainless Steel 304 | 16 | Base-to-Column & Column-to-Neck attachment |
| **M4 × 25 mm Button Head Screws**| M4 × 25 mm Stainless Steel 304 | 8 | Base & Column longitudinal seam clamping |
| **M3 × 12 mm Button Head Screws**| M3 × 12 mm Stainless Steel 304 | 16 | Head perimeter assembly & screen brackets |
| **M3 × 8 mm Socket Head Screws** | M3 × 8 mm Stainless Steel 304 | 8 | Servo mounting ears & Camera PCB |
| **608ZZ Ball Bearings** | 8 mm ID × 22 mm OD × 7 mm Width | 2 | Neck pivot dual bearing suspension |
| **Steel Hinge Dowel Pin / Axle** | Ø8.0 mm × 95.0 mm Polished Steel | 1 | Head pitch rotation axle |
| **M3 Ball-Joint Linkage End** | M3 Rod End Ball Link (Metal) | 1 | Servo pushrod to head bracket |
| **Rubber Anti-Slip Feet** | Ø20 mm × 5 mm Adhesive Rubber Pad | 4 | Base bottom vibration isolation & stability |

---

### C. Electronics & Multimodal Hardware

| Component | Recommended Model | Voltage / Current | Purpose |
| :--- | :--- | :--- | :--- |
| **Main Compute SBC** | Raspberry Pi 5 (8GB) / Jetson Orin Nano / Mini PC | 5V 5A (USB-C) / 19V DC | Runs YOLO, Whisper, Kokoro TTS & Karma Mind |
| **Pitch Servo Motor** | DS3218 (20kg.cm) / DSS-M15S / MG996R | 6.0V – 7.4V @ 2.5A peak | Direct-drive head tilt pitch actuation |
| **Display Screen** | Waveshare 7.0" or 10.1" IPS Capacitive Touch | 5V 1A (HDMI + USB) | Multimodal animated eye UI & visual feedback |
| **Vision Camera** | Wide-Angle 120° FOV USB Camera / Pi Camera v3 | 5V 0.5A (USB / MIPI CSI) | YOLOv8 object detection, face and gaze tracking |
| **Audio Speakers** | Dual 40mm / 50mm 4Ω 5W Full-Range Drivers | 5V Audio Amp (PAM8403 / MAX98357A) | Kokoro-82M speech output |
| **Microphone** | ReSpeaker USB Mic Array / Mini USB Condenser | 5V 0.2A (USB) | Silero VAD & Whisper STT speech recognition |
| **Power Supply / Battery** | 12V 5A AC-DC Adapter OR 12V 6000mAh LiFePO4 | 12V DC input | Main robot power rail |
| **Buck Converter (Step-Down)**| 12V to 5V 5A High-Efficiency DC-DC (LM2596 / XL4015) | 5V 5A output | Clean power for Compute, Display, Servo & Audio |

---

## 3. Slicing & 3D Printing Production Parameters

### Recommended Slicers:
- **Bambu Studio / OrcaSlicer / PrusaSlicer / Cura**

### Optimal Print Settings by Material:

```
┌─────────────────────────┬────────────────────────────────────────────────────────┐
│ Parameter               │ Recommended Value                                      │
├─────────────────────────┼────────────────────────────────────────────────────────┤
│ Material                │ PETG (Recommended), ABS/ASA, or Tough PLA             │
│ Nozzle Temperature      │ 240°C – 250°C (PETG) / 215°C (Tough PLA)               │
│ Bed Temperature         │ 80°C (PETG) / 60°C (Tough PLA) / 100°C (ABS)           │
│ Layer Height            │ 0.20 mm (Optimal strength & detail)                    │
│ Wall Line Count         │ 4 to 6 perimeters (Min 1.6 mm solid shell)             │
│ Top & Bottom Layers     │ 5 top layers, 5 bottom layers (1.0 mm solid caps)      │
│ Infill Density          │ 25% – 40% (Structural parts) / 100% (Linkage arm)      │
│ Infill Pattern          │ Gyroid (Uniform 3D isotropic load distribution)        │
│ Support Type            │ Tree / Organic Supports (Touching buildplate only)     │
│ Brim                    │ 5 mm outer brim (Recommended for Column parts)         │
└─────────────────────────┴────────────────────────────────────────────────────────┘
```

### Build Plate Orientation:
1. **Base Halves (`base_front.stl`, `base_back.stl`)**: Print resting flat on their large bottom faces (Z=0).
2. **Column Halves (`column_front.stl`, `column_back.stl`)**: Print vertically resting on their bottom flange interface.
3. **Neck Halves (`neck_front.stl`, `neck_back.stl`)**: Print resting on their bottom flat mounting plate (Z=0).
4. **Head Bezel (`head_window_half.stl`)**: Print with front screen face flat on the textured PEI bed.
5. **Head Cover (`head_cover_half.stl`)**: Print with perimeter parting plane down against the bed.
6. **Steering Arm (`steering_arm.stl`)**: Print flat on its side with 100% infill for maximum tensile rigidity.

---

## 4. Step-by-Step Assembly Instructions

### Phase 1: Heat-Set Threaded Insert Installation
1. Using a soldering iron with a conical heat-set insert tip heated to **230°C (for PETG)**:
   - Insert **24x M4 brass inserts** into all designated cylindrical mounting bosses in `base_front`, `base_back`, `column_front`, `column_back`, and `neck_front`.
   - Insert **28x M3 brass inserts** into the compute sled rails, servo cradle ears, camera PCB posts, and head perimeter bosses.
2. Press inserts flush with the surface. Allow 60 seconds to cool and solidify into the polymer matrix.

---

### Phase 2: Base Chassis & Power Assembly
1. Mount the 4x rubber anti-slip feet into the bottom alignment pockets of `base_front` and `base_back`.
2. Secure the 12V power supply / battery pack and 12V-to-5V DC-DC buck converter into the rear base bay using M3 screws.
3. Fasten the compute board (Raspberry Pi 5 / Jetson / Mini PC) onto the floor mounting sled using 4x M3×6mm screws.
4. Join `base_front` and `base_back` along the Y=0 seam and tighten the 4x M4×25mm transverse clamping bolts.

---

### Phase 3: Column / Torso & Acoustic Integration
1. Mount the dual 40mm/50mm speaker drivers into the sealed acoustic chambers of `column_front` using 8x M3 screws and foam sealing tape.
2. Mount the microphone array module into the top microphone port at Z=450mm.
3. Route the speaker leads, microphone USB cable, and power lines through the internal central wiring conduit.
4. Mount the assembled Column onto the Base top flange at Z=220mm and secure from inside using 8x M4×16mm socket head screws.

---

### Phase 4: Neck Actuation & Servo Installation
1. Press the **dual 608ZZ ball bearings** into the left and right bearing pockets of `neck_front` and `neck_back`.
2. Place the 20kg digital servo into the neck servo cradle and fasten its mounting ears using 4x M3×8mm screws.
3. Attach the metal 25T servo horn to the servo spline and connect the `steering_arm.stl` linkage using an M3 ball-joint screw.
4. Bolt `neck_front` and `neck_back` together and mount the neck assembly onto the top of the Column at Z=700mm using 4x M4×16mm screws.

---

### Phase 5: Head Display, Camera & Perception Housing
1. Mount the wide-angle HD camera module into the top camera port of `head_window_half.stl` and fasten with 4x M2.5/M3 screws.
2. Seat the 7"/10.1" IPS touchscreen display against the front bezel window and secure with the internal clamping brackets.
3. Connect the HDMI/DSI display ribbon, camera USB/CSI cable, and touch controller through the rear opening.
4. Align the head mounting clevis ears with the Neck 608ZZ bearings and slide the **Ø8mm polished steel dowel pin** through both bearings.
5. Attach the `steering_arm` pushrod linkage to the head bracket.
6. Fasten `head_cover_half.stl` to the front bezel using 8x M3×12mm perimeter screws.

---

### Phase 6: Electrical Hookup & Kinematics Calibration
1. Connect the servo signal lead to GPIO/PWM (Pin 18 / PWM0 on Raspberry Pi).
2. Power on the 12V main switch. Verify 5.1V rail on compute and 6.5V rail on servo.
3. Test pitch angle range:
   - **90° (PWM ~1500µs):** Head faces horizontally forward (level eye contact).
   - **135° (PWM ~1900µs):** Head tilts 45° downward (desk interaction / object inspection).
4. Launch Karma:
   ```bash
   python3 src/main.py
   ```
   Karma will calibrate its gaze, initialize sight (YOLOv8 + MediaPipe), voice (Kokoro-82M), and begin multimodal companion interaction!
