
/*
 * content_ai.highlight.patch.js  (non-destructive add-on)
 * Purpose: Keep ALL your current features. Add:
 *  - Real-time highlight on the SEARCH page for users already in the DB
 *  - Works with either userid or username (and keeps an alias map)
 *  - Updates highlights immediately after you review/auto-close a profile
 * Load this file AFTER your existing content_ai.js.
 */
(function(){
  'use strict';
  const VER = 'highlight-patch v1.5';

  /************** helpers **************/
  const $  = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
  const wait = (ms)=>new Promise(r=>setTimeout(r,ms));

  const parseQS = (u)=> new URL(u || location.href);
  const isSearchLike = ()=> {
    const href = String(location.href || '');
    if (document.querySelector('#search-input, input[type="search"]')) return true;
    return href.includes('/search') || href.includes('keyword=') || href.includes('q=');
  };
  const isProfile = ()=> {
    const p = String(location.pathname || '');
    return p.includes('/profile/') || p.includes('/user/profile/');
  };

  function safeParseJSON(s, fall){ try{ return JSON.parse(s); }catch{ return fall; } }
  function loadSet(key){
    const a = safeParseJSON(localStorage.getItem(key), []);
    return new Set(Array.isArray(a) ? a : []);
  }
  function saveSet(key, set){ localStorage.setItem(key, JSON.stringify(Array.from(set || []))); }

  const LS_ALIAS = 'xhs_user_alias_map'; // { userid: username }
  function loadAlias(){ return safeParseJSON(localStorage.getItem(LS_ALIAS), {}); }
  function saveAlias(map){ localStorage.setItem(LS_ALIAS, JSON.stringify(map || {})); }

  // broadcast (optional)
  let bc=null; try{ bc=new BroadcastChannel('xhs-audit-bc'); }catch{}
  function broadcastReviewed(u){
    try{ bc && bc.postMessage({type:'reviewed', ...u}); }catch{}
  }
  if (bc){
    bc.onmessage = (ev)=>{
      const m = ev.data || {};
      if (m.type==='reviewed' && (m.username || m.userid)){
        if (m.username) sessionNames.add(m.username);
        if (m.userid) { const id=String(m.userid); sessionIds.add(id); if (m.username) { alias[id]=m.username; saveAlias(alias);} }
        scheduleScan();
      }
    };
  }

  /************** known sets **************/
  // pull from your script if present; else, from localStorage
  const sessionNames = (window.sessionMemory?.usernames) || loadSet('xhs_audit_session_usernames');
  const sessionIds   = (window.sessionMemory?.userids)   || loadSet('xhs_audit_session_userids');
  const alias = loadAlias(); // userid -> username

  let knownNames = new Set();
  let knownIds   = new Set();

  async function loadBackendSetsOnce(){
    const urlNames = window.FLASK_URL_USERNAMES || '/api/known_usernames';
    const urlIds   = window.FLASK_URL_USERIDS   || '/api/known_userids';
    try{
      const [n, i] = await Promise.all([
        fetch(urlNames, {credentials:'include'}).then(r=>r.ok?r.json():null).catch(()=>null),
        fetch(urlIds,   {credentials:'include'}).then(r=>r.ok?r.json():null).catch(()=>null),
      ]);
      knownNames = new Set(Array.isArray(n)?n:[]);
      knownIds   = new Set((Array.isArray(i)?i:[]).map(String));
    }catch{}
  }

  function inKnown({username, userid}){
    const id = String(userid||'').trim();
    const name = String(username||'').trim();
    if (id && (knownIds.has(id) || sessionIds.has(id))) return true;
    if (name && (knownNames.has(name) || sessionNames.has(name))) return true;
    // alias: if id known and alias has a username, or name matches alias values
    if (id && alias[id]) return true;
    if (name && Object.values(alias).includes(name)) return true;
    return false;
  }

  /************** card extraction & highlight **************/
  const STYLE_ID = 'xhs-reviewed-highlight-style';
  function injectStyle(){
    if (document.getElementById(STYLE_ID)) return;
    const css = document.createElement('style');
    css.id = STYLE_ID;
    css.textContent = `
      .xhs-reviewed-badge{position:absolute;top:8px;left:8px;z-index:9;
        background:rgba(30,158,90,.95);color:#fff;border-radius:8px;padding:2px 8px;
        font:12px/20px system-ui,Segoe UI,Arial}
      .xhs-reviewed-outline{outline:3px solid rgba(30,158,90,.75);outline-offset:2px;border-radius:12px}
    `;
    document.head.appendChild(css);
  }

  function extractCardInfo(a){
    try{
      const url = new URL(a.href, location.origin);
      const m = url.pathname.match(/\/(?:user\/)?profile\/([^/?#]+)/);
      const userid = m ? m[1] : '';
      // try username from nearby
      let username = (a.getAttribute('title') || a.textContent || '').trim();
      if (!username){
        const card = a.closest('[data-note-card],[data-v-], .note-item, .browse-feed, .feeds-card, .note-card') || a.parentElement;
        username = (card && (card.querySelector('[data-author-name], [class*="author"]')?.textContent || '').trim()) || '';
      }
      return { userid, username };
    }catch{ return {userid:'', username:''}; }
  }

  function cardRoot(a){
    return a.closest('[data-note-card],[data-v-], .note-item, .browse-feed, .feeds-card, .note-card') || a.parentElement || a;
  }

  function markCard(a, reason){
    const root = cardRoot(a);
    if (!root || root.__xhsReviewedMarked) return;
    root.__xhsReviewedMarked = true;
    root.classList.add('xhs-reviewed-outline');
    // badge
    const badge = document.createElement('div');
    badge.className = 'xhs-reviewed-badge';
    badge.textContent = '已在库';
    badge.title = reason || '已在库（去重）';
    root.style.position = root.style.position || 'relative';
    root.appendChild(badge);
  }

  let scanScheduled = false;
  function scheduleScan(){ if (!scanScheduled){ scanScheduled=true; requestAnimationFrame(runScan); } }

  function runScan(){
    scanScheduled=false;
    if (!isSearchLike()) return;
    injectStyle();
    const anchors = $$('a[href*="/profile/"], a[href*="/user/profile/"]');
    for (const a of anchors){
      if (a.__xhsReviewedChecked) continue;
      a.__xhsReviewedChecked = true;
      const info = extractCardInfo(a);
      if (!info.userid && !info.username) continue;
      if (inKnown(info)){
        markCard(a, 'in DB');
      }
    }
  }

  // MutationObserver to catch infinite-scroll
  let mo = null;
  function startObserver(){
    if (mo || !isSearchLike()) return;
    mo = new MutationObserver(()=> scheduleScan());
    mo.observe(document.body, {childList:true, subtree:true});
  }

  // Profile page: store alias (userid -> username) and broadcast so search page can highlight immediately
  async function onProfile(){
    const u = parseQS();
    const id = (u.pathname.split('/').pop() || '').trim();
    let username = '';
    const h = document.querySelector('h1, h2'); if (h) username = (h.textContent||'').trim();
    if (!username){
      const og = document.querySelector('meta[property="og:title"][content]');
      if (og) username = (og.getAttribute('content')||'').trim();
    }
    if (id){
      alias[id] = alias[id] || username || alias[id] || '';
      saveAlias(alias);
      if (username){
        // write-through to session memory if available
        try{
          window.sessionMemory?.userids?.add?.(id);
          window.sessionMemory?.usernames?.add?.(username);
        }catch{}
      }
      broadcastReviewed({userid:id, username});
    }
  }

  // Boot
  (async function boot(){
    await loadBackendSetsOnce();
    if (isSearchLike()){ startObserver(); scheduleScan(); }
    if (isProfile()){ onProfile(); }
    console.debug('[%s] ready.', VER);
  })();
})();
