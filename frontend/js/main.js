import { initTooltips, initReveal, initNav } from "./util.js";
import { initStudio } from "./studio.js";
import { initLibrary } from "./library.js";
import { initPipeline } from "./pipeline.js";
import { initResearch } from "./research.js";
import { initSamples } from "./samples.js";
import { initShowcase } from "./showcase.js";
import { initVideos } from "./videos.js";
import { initAuth, adminReady } from "./auth.js";
import { initAdmin } from "./admin.js";

initAuth().then(() => {
  initNav();
  initTooltips();
  initPipeline();
  initStudio();
  initLibrary();
  initResearch().finally(initReveal);
  initSamples();
  initShowcase();
  initVideos();
  adminReady.then(initAdmin);
});
