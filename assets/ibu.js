/* ibu: scroll reveal, count-up, header filter bar. Dash loads every file in /assets. */
(function () {
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function fmt(n) { return n.toLocaleString('en-US'); }
  function count(el) {
    var to = +el.getAttribute('data-count'), suf = el.getAttribute('data-suffix') || '';
    if (reduce || !to) { el.textContent = fmt(to) + suf; return; }
    var t0 = null, dur = 1600;
    function step(t) {
      if (!t0) t0 = t;
      var p = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(Math.round(to * e)) + suf;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  var io = 'IntersectionObserver' in window ? new IntersectionObserver(function (es) {
    es.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add('in');
      e.target.querySelectorAll('[data-count]').forEach(count);
      io.unobserve(e.target);
    });
  }, { threshold: 0.08 }) : null;
  function scan() {
    document.querySelectorAll('.reveal:not([data-obs])').forEach(function (el) {
      el.setAttribute('data-obs', '1');
      if (io) io.observe(el); else el.classList.add('in');
    });
  }
  function onScroll() {
    var h = document.getElementById('header'), l = document.getElementById('landing');
    if (h && l) h.classList.toggle('with-filter', l.getBoundingClientRect().bottom < 160);
    else if (h) h.classList.add('with-filter');   /* no landing page: always show the slider */
  }
  setInterval(scan, 300);
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('load', onScroll);
})();
