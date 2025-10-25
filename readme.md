# Real-time Mood Monitor

A web app that monitors and displays your mood **in real time** using facial emotion recognition via webcam.  
Built with **Vite**, **TensorFlow.js**, and modern web technologies.

---

## Demo

**Live App:** [Real-time Mood Monitor](https://real-time-mood-monitor.vercel.app)  
**Source Code:** [GitHub Repo](https://github.com/zameernagaral/Real-time-Mood-Monitor)

---

## Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## Features

- Real-time **emotion / mood detection** from webcam.
- Visual overlay showing current mood.
- Works directly in browser — no server needed.
- Fast performance using **TensorFlow.js**.
- Modular and easy to extend.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| Framework / Bundler | Vite |
| Language | JavaScript (ES6+), HTML, CSS |
| ML Library | TensorFlow.js / MobileNet |
| Deployment | Vercel |
| Browser APIs | `getUserMedia` (Webcam) |

---

## How It Works

1. The app captures live video using your webcam.  
2. A pre-trained model (like MobileNet) detects facial features and emotions.  
3. Based on classification results, it displays your **mood in real time**.  
4. The UI updates continuously as you change expressions.

> **Note:** This project is for **learning and fun**, not a diagnostic tool.

---

## Setup & Installation

Clone the repository and run locally:

```bash
# Clone repo
git clone https://github.com/zameernagaral/Real-time-Mood-Monitor.git

# Navigate into folder
cd Real-time-Mood-Monitor

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
