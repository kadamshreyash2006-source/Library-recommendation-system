/* Shared front-end helpers for the Smart Library site:
   covers, book tiles, star ratings, save-to-library, title suggestions. */

const LIB = window.LIB = {
  authed: document.body.dataset.authed === '1',
  shelfOrder: JSON.parse(document.body.dataset.shelves || '[]'),   // [[key, label], ...] in display order
};
LIB.shelves = Object.fromEntries(LIB.shelfOrder);

function esc(s){
  return String(s ?? '').replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
window.esc = esc;

/* ---------- API ---------- */
async function api(path, body){
  const opts = body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)};
  const r = await fetch(path, opts);
  let d = null;
  try { d = await r.json(); } catch(e) {}
  if (r.status === 401){ goLogin(); throw new Error('login'); }
  if (!r.ok) throw new Error((d && d.error) || 'Something went wrong.');
  return d;
}
window.api = api;

function goLogin(){
  location.href = '/login?next=' + encodeURIComponent(location.pathname + location.search);
}

/* ---------- toast ---------- */
let toastTimer = null;
function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
}
window.toast = toast;

/* ---------- covers: Open Library by ISBN, then publisher URL, then title ---------- */
function coverHTML(b, size){
  const cls = 'cover-' + size, phc = 'ph-' + size;
  const srcs = [];
  if (b.isbn) srcs.push('https://covers.openlibrary.org/b/isbn/' + encodeURIComponent(b.isbn) + '-M.jpg?default=false');
  if (b.cover_url && b.cover_url !== 'nan') srcs.push(b.cover_url);
  const short = esc(b.title).split(/\s+/).slice(0, size === 'sm' ? 4 : 7).join(' ');
  if (!srcs.length) return `<div class="cover ${cls} ph ${phc}">${short}</div>`;
  return `<img class="cover ${cls}" src="${esc(srcs[0])}" alt="" loading="lazy"
    data-srcs='${esc(JSON.stringify(srcs))}' data-i="0" data-short="${short}" data-ph="${phc}"
    onerror="nextCover(this)">`;
}
function nextCover(img){
  let list = [];
  try { list = JSON.parse(img.dataset.srcs); } catch(e) {}
  const i = parseInt(img.dataset.i || '0', 10) + 1;
  if (i < list.length){ img.dataset.i = i; img.src = list[i]; return; }
  const ph = document.createElement('div');
  ph.className = img.className + ' ph ' + (img.dataset.ph || 'ph-sm');
  ph.textContent = img.dataset.short || '';
  img.replaceWith(ph);
}
window.coverHTML = coverHTML;
window.nextCover = nextCover;

/* ---------- stars & shelf controls ---------- */
function starsHTML(b, big){
  const r = b.user_rating || 0;
  let h = `<span class="stars${big ? ' lg' : ''}" data-index="${b.index}" data-rating="${r}" role="group" aria-label="Your rating">`;
  for (let s = 5; s >= 1; s--){
    h += `<button type="button" data-star="${s}" class="${s <= r ? 'on' : ''}" aria-label="${s} star${s > 1 ? 's' : ''}" title="${s} star${s > 1 ? 's' : ''}">★</button>`;
  }
  return h + '</span>';
}
function shelfHTML(b, big){
  const cur = b.shelf || '';
  let h = `<select class="shelf${cur ? ' saved' : ''}${big ? ' lg' : ''}" data-index="${b.index}" aria-label="Save to library">`;
  h += `<option value="" ${cur ? '' : 'selected'}>${cur ? 'Move to…' : '＋ Save to library'}</option>`;
  for (const [k, label] of LIB.shelfOrder){
    h += `<option value="${k}" ${cur === k ? 'selected' : ''}>${cur === k ? '✓ ' : ''}${esc(label)}</option>`;
  }
  if (cur) h += `<option value="__remove">Remove from library</option>`;
  return h + '</select>';
}
window.starsHTML = starsHTML;
window.shelfHTML = shelfHTML;

// After the server answers, refresh every control for that book on the page.
function syncBook(b){
  document.querySelectorAll(`.stars[data-index="${b.index}"]`).forEach(el => {
    el.outerHTML = starsHTML(b, el.classList.contains('lg'));
  });
  document.querySelectorAll(`.shelf[data-index="${b.index}"]`).forEach(el => {
    el.outerHTML = shelfHTML(b, el.classList.contains('lg'));
  });
  document.dispatchEvent(new CustomEvent('book-updated', {detail:b}));
}
window.syncBook = syncBook;

document.addEventListener('click', async e => {
  const btn = e.target.closest('.stars button');
  if (!btn) return;
  e.preventDefault();
  if (!LIB.authed) return goLogin();
  const wrap = btn.parentElement;
  const index = parseInt(wrap.dataset.index, 10);
  let star = parseInt(btn.dataset.star, 10);
  if (star === parseInt(wrap.dataset.rating, 10)) star = 0;   // click again to clear
  try{
    const b = await api('/api/rate', {index, rating:star});
    syncBook(b);
    toast(star ? `Rated ${star} star${star > 1 ? 's' : ''}. Your recommendations will adapt.` : 'Rating removed.');
  }catch(err){ if (err.message !== 'login') toast(err.message); }
});

document.addEventListener('change', async e => {
  const sel = e.target.closest('select.shelf');
  if (!sel) return;
  if (!LIB.authed) return goLogin();
  const index = parseInt(sel.dataset.index, 10);
  const v = sel.value;
  if (!v) return;
  try{
    let b;
    if (v === '__remove'){
      b = await api('/api/library/remove', {index});
      toast('Removed from your library.');
    } else {
      b = await api('/api/library', {index, shelf:v});
      toast('Saved to ' + LIB.shelves[v] + '.');
    }
    syncBook(b);
  }catch(err){ if (err.message !== 'login') toast(err.message); }
});

/* ---------- book tile for grids ---------- */
function tileHTML(b, opts = {}){
  const why = opts.reason && b.reason ? `<div class="why">${esc(b.reason.text)}</div>` : '';
  const badge = opts.badge ? `<span class="badge">${esc(opts.badge)}</span>` : '';
  return `<div class="tile" data-book="${b.index}">
    <a class="cover-link" href="/book/${b.index}" aria-label="${esc(b.title)}">${coverHTML(b, 'md')}${badge}</a>
    <a class="t" href="/book/${b.index}">${esc(b.title)}</a>
    <div class="a">${esc(b.author)}</div>
    ${why}
    <div class="actions">${starsHTML(b)}${shelfHTML(b)}</div>
  </div>`;
}
window.tileHTML = tileHTML;

function skeletonGrid(n){
  let h = '';
  for (let i = 0; i < n; i++) h += '<div class="tile skeleton"><div class="cover cover-md"></div><div class="sk-line"></div><div class="sk-line s"></div></div>';
  return h;
}
window.skeletonGrid = skeletonGrid;

/* ---------- title suggestions (used by nav search and discover) ---------- */
function attachSuggest(input, box, onPick, onEnter){
  let items = [], active = -1, timer = null;
  const close = () => { box.classList.remove('open'); active = -1; };
  const show = list => {
    items = list;
    if (!list.length) return close();
    box.innerHTML = list.map((s, i) =>
      `<div class="sug" data-i="${i}"><div class="t">${esc(s.title)}</div><div class="a">${esc(s.author)}</div></div>`).join('');
    box.classList.add('open');
  };
  box.addEventListener('mousedown', e => {
    const el = e.target.closest('.sug');
    if (!el) return;
    e.preventDefault();
    const s = items[parseInt(el.dataset.i, 10)];
    close(); onPick(s);
  });
  input.addEventListener('input', () => {
    input.dataset.chosen = '';
    clearTimeout(timer);
    const v = input.value.trim();
    if (v.length < 2) return close();
    timer = setTimeout(async () => {
      try { show(await (await fetch('/api/search?q=' + encodeURIComponent(v))).json()); }
      catch(e){ close(); }
    }, 170);
  });
  input.addEventListener('keydown', e => {
    if (!box.classList.contains('open')){
      if (e.key === 'Enter'){ e.preventDefault(); onEnter && onEnter(input.value.trim()); }
      return;
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp'){
      e.preventDefault();
      active += e.key === 'ArrowDown' ? 1 : -1;
      if (active < 0) active = items.length - 1;
      if (active >= items.length) active = 0;
      box.querySelectorAll('.sug').forEach((el, i) => el.classList.toggle('active', i === active));
    } else if (e.key === 'Enter'){
      e.preventDefault();
      if (active >= 0){ const s = items[active]; close(); onPick(s); }
      else { close(); onEnter && onEnter(input.value.trim()); }
    } else if (e.key === 'Escape') close();
  });
  input.addEventListener('blur', () => setTimeout(close, 120));
  return {close};
}
window.attachSuggest = attachSuggest;

/* ---------- nav ---------- */
(function(){
  const inp = document.getElementById('nav-q');
  if (inp){
    attachSuggest(inp, document.getElementById('nav-suggest'),
      s => { location.href = '/book/' + s.index; },
      v => { if (v) location.href = '/discover?q=' + encodeURIComponent(v); });
  }
  const tg = document.getElementById('menu-toggle');
  if (tg) tg.onclick = () => document.getElementById('nav-links').classList.toggle('open');
})();
