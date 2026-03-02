const canvas = document.getElementById("bg-canvas");
const ctx = canvas.getContext("2d");
let width = 0;
let height = 0;

const nodes = Array.from({ length: 64 }, () => ({
  x: 0,
  y: 0,
  vx: 0,
  vy: 0,
  r: 0
}));

function resetNode(node) {
  node.x = Math.random() * width;
  node.y = Math.random() * height;
  node.vx = (Math.random() - 0.5) * 0.36;
  node.vy = (Math.random() - 0.5) * 0.36;
  node.r = 0.4 + Math.random() * 1.8;
}

function resize() {
  width = canvas.width = window.innerWidth;
  height = canvas.height = window.innerHeight;
  nodes.forEach(resetNode);
}

function tick() {
  ctx.clearRect(0, 0, width, height);

  for (let i = 0; i < nodes.length; i += 1) {
    const a = nodes[i];
    a.x += a.vx;
    a.y += a.vy;

    if (a.x < 0 || a.x > width || a.y < 0 || a.y > height) {
      resetNode(a);
      continue;
    }

    for (let j = i + 1; j < nodes.length; j += 1) {
      const b = nodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 130) {
        const alpha = 0.09 * (1 - dist / 130);
        ctx.strokeStyle = `rgba(100, 214, 255, ${alpha})`;
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    ctx.fillStyle = "rgba(138, 255, 200, 0.8)";
    ctx.beginPath();
    ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2);
    ctx.fill();
  }

  requestAnimationFrame(tick);
}

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.14 });

document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));

window.addEventListener("resize", resize);
resize();
tick();
