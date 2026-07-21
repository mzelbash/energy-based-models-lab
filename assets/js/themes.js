// Theme switcher. Three themes: light, dark, projector. The choice persists in
// localStorage and falls back to the operating system preference on first visit.
// No long dashes are used anywhere in this project.

const KEY = 'ebm-theme';
const THEMES = ['light', 'dark', 'projector'];

export function initThemes() {
  const seg = document.getElementById('theme-seg');
  const stored = localStorage.getItem(KEY);
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const initial = stored && THEMES.includes(stored) ? stored : prefersDark ? 'dark' : 'light';
  apply(initial);

  if (seg) {
    seg.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => apply(btn.dataset.theme));
    });
  }

  function apply(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(KEY, theme);
    if (seg) {
      seg.querySelectorAll('button').forEach((b) =>
        b.setAttribute('aria-pressed', String(b.dataset.theme === theme))
      );
    }
    // let interested views (for example canvas charts) recolor
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
  }
}
