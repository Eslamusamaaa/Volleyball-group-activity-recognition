import { initTooltips, initReveal, initNav } from "./util.js";
import { initStudio } from "./studio.js";
import { initLibrary } from "./library.js";
import { initPipeline } from "./pipeline.js";
import { initResearch } from "./research.js";
import { initSamples } from "./samples.js";
import { initShowcase } from "./showcase.js";
import { initVideos } from "./videos.js";
import { initAuth } from "./auth.js";
import { initAdmin } from "./admin.js";
import { initTrack } from "./track.js";

// The site opens straight away. Sign-in is optional and happens from inside the site (nav button, Studio).
initNav();
initTooltips();
initPipeline();
initStudio();
initLibrary();
initResearch().finally(initReveal);
initSamples();
initShowcase();
initVideos();
initAdmin();    // listens for the sign-in event before auth can emit it
initAuth();
initTrack();
