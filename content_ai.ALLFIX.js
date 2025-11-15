// content_ai.js (v10.21 - 核心重构最终版)
// - 【基准】本代码严格基于 v10.14 完整版本，并整合 v10.20 的调试日志。
// - 【重复问题根源修复】引入独立的 `sessionMemory` Set。syncFromBackend 只在任务开始时向其填充一次数据。后续所有批次的用户都只向 sessionMemory 中添加，彻底切断了被后端旧数据覆盖的可能性，根除重复打开问题。
// - 【关键词BUG修复】废弃不稳定的 storage 缓存同步。改为在打开个人主页时，将当前搜索词作为 URL 参数 (`&search_term=...`) 直接附加在链接上。个人主页从 URL 中直接读取，确保 100% 准确无误，零延迟。
// - 【卡死BUG修复】严格区分模式：在“AI建议模式”（人工审核）下，打开一轮页面后，任务立即结束并提示成功，【绝不调用】`waitForBatchBySignal` 等待函数，彻底解决任务卡死问题。
// - 【完整性】保留 v10.14 的所有功能，未做任何删减。

(function () {
  'use strict';

// === expose backend endpoints to window (for patches/highlight/etc.) ===
try {
  if (typeof FLASK_BASE === 'string')                   window.FLASK_BASE = FLASK_BASE;
  if (typeof FLASK_MARK_URL === 'string')               window.FLASK_MARK_URL = FLASK_MARK_URL;
  if (typeof FLASK_URL_USERNAMES === 'string')          window.FLASK_URL_USERNAMES = FLASK_URL_USERNAMES;
  if (typeof FLASK_URL_USERIDS === 'string')            window.FLASK_URL_USERIDS = FLASK_URL_USERIDS;
  if (typeof FLASK_AI_SUGGEST_URL === 'string')         window.FLASK_AI_SUGGEST_URL = FLASK_AI_SUGGEST_URL;
  if (typeof FLASK_AI_DECIDE_URL === 'string')          window.FLASK_AI_DECIDE_URL = FLASK_AI_DECIDE_URL;
  if (typeof FLASK_SAVE_HISTORY_SETTINGS_URL === 'string') window.FLASK_SAVE_HISTORY_SETTINGS_URL = FLASK_SAVE_HISTORY_SETTINGS_URL;
} catch (e) {}


  const FLASK_BASE = "http://localhost:5001";
  
  const FLASK_MARK_URL            = `${FLASK_BASE}/mark_data`;
  const FLASK_URL_USERNAMES       = `${FLASK_BASE}/usernames`;
  const FLASK_URL_USERIDS         = `${FLASK_BASE}/userids`;
  const FLASK_AI_DECIDE_URL       = `${FLASK_BASE}/ai/decide`;
  const FLASK_AI_SUGGEST_URL      = `${FLASK_BASE}/ai/suggest`;
  const FLASK_AI_SETTINGS_URL     = `${FLASK_BASE}/ai/settings`;
  const FLASK_SAVE_HISTORY_SETTINGS_URL = `${FLASK_BASE}/settings/save_history`;
  const FLASK_EXPORT_URL          = `${FLASK_BASE}/export_delta?dataset=approved`;
  const FLASK_GET_REVIEW_LIST_URL = `${FLASK_BASE}/get_review_list`;
  const FLASK_DASHBOARD_URL       = `${FLASK_BASE}/dashboard`;

  const highlightStyle = "background-color: yellow !important; color: red !important; font-weight: bold !important;";
  const OPEN_BATCH_LIMIT = 15;
  const HUMAN_MIN_DELAY_MS = 800;
  const HUMAN_MAX_DELAY_MS = 2500;
  const AUTOSCROLL_STEP = 1400;
  const AUTOSCROLL_ROUND_WAIT = 550;
  const AUTOSCROLL_IDLE_ROUNDS = 4;
  
  const FOLLOWER_SELECTOR = "#userPageContainer > div.user > div > div.info-part > div.info > div.data-info > div > div:nth-child(2) > span.count";
  const TOTAL_LIKES_SELECTORS = [ "#userPageContainer > div.user > div > div.info-part > div.info > div.data-info > div > div:nth-child(3) > span.count", "#userPageContainer > div.user > div > div.info-part > div.info > div.data-info > div > div:nth-child(1) > span.count" ];
  const BIO_SELECTOR_PREF = "#userPageContainer > div.user > div > div.info-part > div.info > div.user-desc";
  const noteTitleSelByIndex = (i) => `#userPostedFeeds > section:nth-child(${i}) > div > div > a > span`;
  const noteCoverSelByIndex = (i) => `#userPostedFeeds > section:nth-child(${i}) > div > a.cover.mask.ld > img`;
  const noteLikeSelByIndex  = (i) => `#userPostedFeeds > section:nth-child(${i}) > div > div > div > span > span.count`;

  let globalAiEnabled = false;
  let globalSaveHistoryEnabled = false;
  const auditTaskState = { isRunning: false, shouldStop: false };
  let pageDetectorInterval = null;

  // ========== 【重复问题根源修复】 ==========
  // 引入独立的会话内存，与全局的 knownUsernames/knownUserids 分离
  const sessionMemory = {
      usernames: new Set(),
      userids: new Set()
  };

  const sleep = (ms) => new Promise(r=>setTimeout(r, ms));
  function extAlive(){ try { return !!(chrome && chrome.runtime && chrome.runtime.id); } catch { return false; } }
  function safeSendMessage(msg){ return new Promise((resolve)=> { if (!extAlive()) return resolve({ ok:false, err:"invalidated" }); try { chrome.runtime.sendMessage(msg, (res)=>{ const err = chrome.runtime.lastError; if (err) { return resolve({ ok:false, err:err.message }); } resolve(res || { ok:true }); }); } catch(e){ resolve({ ok:false, err:String(e) }); } }); }
  function safeStorageGet(keys){ return new Promise((resolve)=>{ if (!extAlive()) return resolve({}); try { chrome.storage.local.get(keys, resolve); } catch { resolve({}); } }); }
  function safeStorageSet(obj){ return new Promise((resolve)=>{ if (!extAlive()) return resolve(false); try { chrome.storage.local.set(obj, ()=> resolve(!chrome.runtime.lastError)); } catch { resolve(false); } }); }
  function requestCloseTab(){ safeSendMessage({ action:"closeTab", url: window.location.href }); }
  function safeLog(prefix, ...args){ try{ console.log(prefix, ...args);}catch(e){} }
  async function GET_JSON(url){ try{ const r=await fetch(url,{credentials:"omit"}); if(!r.ok) return null; return r.json(); } catch{ return null; } }
  async function postJSON(url, data){
  try {
    return fetch(url, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(data || {}),
      credentials: 'omit'
    }).then(res => res.ok ? res.json().catch(()=>true) : null)
      .catch(() => null);
  } catch(e){
    console.warn('postJSON error', url, e);
    return Promise.resolve(null);
  }
}, body:JSON.stringify(obj)}); if (!r.ok) return { ok: false, error: `HTTP ${r.status}` }; return r.json(); } catch(e) { return { ok: false, error: String(e) }; } }
  function isProfilePage(){ return /\/user\/profile\//i.test(location.pathname); }
  function isSearchPage(){ return /\/explore|\/search|\/explore\?/i.test(location.pathname); }
  function updateStatus(text, level="info"){ const el=document.getElementById("xhs-status"); if(!el) return; el.textContent=text; el.style.color = level==="error"?"#ff7676":(level==="success"?"#a6ffa6":(level==="warn"?"#ffdd99":"#ccc")); }
  const debounce = (func, wait) => { let timeout; return (...args) => { clearTimeout(timeout); timeout = setTimeout(() => func.apply(this, args), wait); }; };
  
  function grabData(){ const nameEl = document.querySelector("#userPageContainer .user-nickname div, #userPageContainer > div.user div.user-nickname > div"); const idEl = document.querySelector("#userPageContainer span.user-redId, #userPageContainer > div.user span.user-redId"); return { username: nameEl?.textContent.trim() || "", userid: idEl?.textContent.replace("小红书号","").replace("：","").trim() || "" }; }
  function scrapeBio(){ const el = document.querySelector(BIO_SELECTOR_PREF); return (el?.textContent||el?.innerText||"").trim(); }
  function parseCountChinese(txt) { if (!txt) return 0; txt = String(txt).trim().replace(/[\uFF10-\uFF19\.]/g, ch => String.fromCharCode(ch.charCodeAt(0) - 0xFF10 + 0x30)).replace(/,/g, "").replace(/\+$/, ""); const mWan = txt.match(/^(\d+(?:\.\d+)?)\s*万$/); if (mWan) return Math.round(parseFloat(mWan[1]) * 10000); const mNum = txt.match(/^(\d+)/); if (mNum) return parseInt(mNum[1], 10) || 0; return 0; }
  function getStats() { const followers = parseCountChinese(document.querySelector(FOLLOWER_SELECTOR)?.textContent); const likes_total_el = TOTAL_LIKES_SELECTORS.map(s => document.querySelector(s)).find(el => el); const likes_total = parseCountChinese(likes_total_el?.textContent); return { followers, likes_total }; }
  function scrapeNotesByTemplate(maxN = 20) { const list = []; for (let i = 1; i <= maxN; i++) { const title = document.querySelector(noteTitleSelByIndex(i))?.textContent?.trim() || ""; const cover = document.querySelector(noteCoverSelByIndex(i))?.src || ""; const likes = parseCountChinese(document.querySelector(noteLikeSelByIndex(i))?.textContent); if (title || cover) list.push({ idx:i, title, cover_url: cover, likes: likes === null ? 0 : likes }); } return list; }
  function highlightElement(el) { if (el && !el.dataset.xhsHighlighted) { el.style.cssText += highlightStyle; el.dataset.xhsHighlighted = "true"; } }
  
  // 这个函数现在只用于个人主页和页面加载时的高亮，不再参与审核循环的记忆
  async function syncKnownDataForHighlighting() {
    const [names, ids] = await Promise.all([GET_JSON(FLASK_URL_USERNAMES), GET_JSON(FLASK_URL_USERIDS)]);
    let knownUsernames = new Set();
    let knownUserids = new Set();
    if (Array.isArray(names)) knownUsernames = new Set(names);
    if (Array.isArray(ids)) knownUserids = new Set(ids);
    
    document.querySelectorAll("a[href*='/user/profile/']").forEach(el => {
        const name = el.querySelector(".name, .user-name")?.textContent.trim();
        const uid = el.getAttribute("data-user-id")?.trim() || el.closest("[data-user-id]")?.getAttribute("data-user-id")?.trim();
        if ((name && knownUsernames.has(name)) || (uid && knownUserids.has(uid))) {
            highlightElement(el.closest("figure, .note-item, section, .user-item"));
        }
    });
  }

  function installMutationObserver(){ const containers = [ document.querySelector("#exploreFeeds"), document.querySelector(".search-layout__main .feeds-container"), document.querySelector("#global .search-layout__main .feeds-container"), document.querySelector(".user-list"), document.querySelector("#user-list"), document.querySelector(".feeds-container") ].filter(Boolean); if (!containers.length) return; const ob = new MutationObserver(() => { clearTimeout(ob._tid); ob._tid = setTimeout(syncKnownDataForHighlighting, 120); }); containers.forEach(c => ob.observe(c, { childList: true, subtree: true })); }
  
  // ========== 【关键词BUG修复】 ==========
  // openTabsSequentially 现在需要接收当前的搜索词
  async function openTabsSequentially(urls, searchTerm, isRereview = false){ 
      const uniqUrls=Array.from(new Set(urls.filter(Boolean)));
      for (const url of uniqUrls) {
          if (auditTaskState.shouldStop) break;
          let finalUrl;
          try {
              const urlObj = new URL(url);
              if (isRereview) {
                  urlObj.searchParams.set('rereview', '1');
              } else {
                  // 将搜索词编码后附加到URL上
                  urlObj.searchParams.set('search_term', encodeURIComponent(searchTerm));
              }
              finalUrl = urlObj.toString();
          } catch(e) {
              console.error("Invalid URL:", url);
              continue;
          }
          await safeSendMessage({ action:"openTab", url: finalUrl });
          await sleep(Math.random() * (HUMAN_MAX_DELAY_MS - HUMAN_MIN_DELAY_MS) + HUMAN_MIN_DELAY_MS);
      }
      return uniqUrls.length;
  }

  function setButtonState(isAuditing) { const startStopBtn = document.getElementById('xhs-start-stop-btn'); if (startStopBtn) { startStopBtn.textContent = isAuditing ? "停止审核" : "开始审核"; startStopBtn.style.background = isAuditing ? "#dc3545" : "#4CAF50"; } }
  function findScrollContainer() { const candidates = [document.scrollingElement, document.documentElement, document.body, document.querySelector('.main-content'), document.querySelector('#app-container')]; for (const el of candidates) { if (el && el.scrollHeight > el.clientHeight + 100) return el; } return document.scrollingElement || document.documentElement; }
  async function autoScrollAndHarvestAll(){ const container=findScrollContainer(); const found=new Map(); let lastSeen=0; let idleRounds=0; for (let round=0; round<500; round++){ if(auditTaskState.shouldStop) break; snapshotCardsInto(found); container.scrollTop += AUTOSCROLL_STEP; await sleep(AUTOSCROLL_ROUND_WAIT); const curCount=found.size; idleRounds=(curCount<=lastSeen)?(idleRounds+1):0; lastSeen=curCount; if (idleRounds >= AUTOSCROLL_IDLE_ROUNDS) break; } container.scrollTop = 0; safeLog("🕵️‍♂️ [审核循环-调试]", "采集完成，总用户数 =", found.size); return Array.from(found.values()); }
  function snapshotCardsInto(map){ document.querySelectorAll("a[href*='/user/profile/']").forEach(link => { const href = (link.href || "").trim(); if (!href) return; const nameElem = link.querySelector(".name, .user-name, .user-nickname, .author-name, .nickname"); const uname = (nameElem?.textContent || "").trim(); const userItem = link.closest('.user-item, section, figure, .note-item, .item, [data-user-id]'); const uid = userItem?.getAttribute("data-user-id") || ""; const key = uid || href; if (!map.has(key)) { map.set(key, { href, uname, uid }); } }); }
  
  // ========== 【重复问题根源修复】 ==========
  // 过滤器现在使用独立的 sessionMemory
  function filterNotReviewed(cards){
      return cards.filter(({uname, uid})=>{
          const u = (uname || "").trim();
          const i = (uid || "").trim();
          if (!u && !i) return false;
          // 使用会话内存进行判断
          return !(sessionMemory.usernames.has(u) || sessionMemory.userids.has(i));
      });
  }

  function waitForBatchBySignal(batchSize) { return new Promise(resolve => { if (batchSize === 0) return resolve(true); let completedCount = 0; updateStatus(`⏳ 等待 ${batchSize} 个页面完成... (0/${batchSize})`, "info"); const listener = (msg, sender, sendResponse) => { if (msg.action === 'task_completed') { completedCount++; updateStatus(`⏳ 等待 ${batchSize} 个页面完成... (${completedCount}/${batchSize})`, "info"); if (completedCount >= batchSize) { chrome.runtime.onMessage.removeListener(listener); updateStatus("✅ 当前批次已完成。", "success"); resolve(true); } } }; chrome.runtime.onMessage.addListener(listener); const timeout = batchSize * 15 * 1000; setTimeout(() => { if (chrome.runtime.onMessage.hasListener(listener)) { chrome.runtime.onMessage.removeListener(listener); console.warn(`等待批次超时(${timeout/1000}s)，强制继续。`); updateStatus("⚠️ 等待超时，可能部分页面未响应", "warn"); resolve(true); } }, timeout); }); }

  // ========== 【核心逻辑重构】 ==========
  async function mainAuditLoop() {
      if (auditTaskState.isRunning) { auditTaskState.shouldStop = true; setButtonState(false); updateStatus("⛔ 正在停止任务...", "warn"); return; }
      Object.assign(auditTaskState, { isRunning: true, shouldStop: false });
      setButtonState(true);
      
      try {
          safeLog("🕵️‍♂️ [审核循环-调试]", "============== 【开始审核】任务启动 ==============");
          updateStatus("⏬ 同步“长期记忆”到会话...", "info");
          
          const [names, ids] = await Promise.all([GET_JSON(FLASK_URL_USERNAMES), GET_JSON(FLASK_URL_USERIDS)]);
          if (names === null || ids === null) throw new Error("后端连接失败，任务中止。");
          
          sessionMemory.usernames = new Set(names);
          sessionMemory.userids = new Set(ids);
          safeLog("🕵️‍♂️ [审核循环-调试]", `[任务启动时] 1. '长期记忆' 已同步至'会话记忆'。用户名: ${sessionMemory.usernames.size}个, ID: ${sessionMemory.userids.size}个`);
          
          let batchNum = 1;
          while (!auditTaskState.shouldStop) {
              safeLog("🕵️‍♂️ [审核循环-调试]", `============== 批次 #${batchNum} 开始 ==============`);
              updateStatus(`⏬ 自动滚动，采集第 ${batchNum} 批新用户...`, "info");
              const allCards = await autoScrollAndHarvestAll();
              safeLog("🕵️‍♂️ [审核循环-调试]", `2. 页面上共采集到 ${allCards.length} 个用户卡片。`);
              
              const candidates = filterNotReviewed(allCards);
              safeLog("🕵️‍♂️ [审核循环-调试]", `3. 使用【会话记忆】过滤后，剩下 ${candidates.length} 个候选用户。`);
              
              if (candidates.length === 0) {
                  updateStatus("✅ 所有可见用户均已审核完毕！", "success");
                  break;
              }

              const batchToOpen = candidates.slice(0, OPEN_BATCH_LIMIT);
              safeLog("🕵️‍♂️ [审核循环-调试]", `4. 本批次将打开 ${batchToOpen.length} 个用户:`, batchToOpen.map(c => c.uname || c.uid).filter(Boolean).join(', '));
              
              for (const user of batchToOpen) {
                  if (user.uname) sessionMemory.usernames.add(user.uname);
                  if (user.uid) sessionMemory.userids.add(user.uid);
              }
              safeLog("🕵️‍♂️ [审核循环-调试]", `5.【关键】加入'短期记忆'后，'会话记忆'更新为: 用户名 ${sessionMemory.usernames.size}个, ID ${sessionMemory.userids.size}个`);
              
              const currentSearchTerm = document.querySelector('.search-input-container input[type="text"]')?.value.trim() || "";
              updateStatus(`🚀 正在打开第 ${batchNum} 批 (共 ${batchToOpen.length} 个)...`, "info");
              const openedCount = await openTabsSequentially(batchToOpen.map(c => c.href), currentSearchTerm);
              
              if (auditTaskState.shouldStop) break;
              
              // ========== 【卡死BUG修复】 ==========
              // 只有在 AI决策模式下 才等待
              if (globalAiEnabled) {
                  await waitForBatchBySignal(openedCount);
                  if (auditTaskState.shouldStop) break;
                  batchNum++;
                  await sleep(1000);
              } else {
                  // AI建议模式（人工审核），不等待，直接结束
                  updateStatus(`✅ 已打开 ${openedCount} 个页面供您审核。`, "success");
                  break;
              }
          }
      } catch (e) {
          updateStatus(`❌ 审核循环出错: ${e.message}`, "error");
          console.error(e);
      } finally {
          safeLog("🕵️‍♂️ [审核循环-调试]", "============== 【开始审核】任务结束 ==============");
          if (auditTaskState.shouldStop) updateStatus("⛔ 任务已手动停止。", "warn");
          Object.assign(auditTaskState, { isRunning: false, shouldStop: false });
          setButtonState(false);
      }
  }

  async function markData(manualStatus, closePage = true) {
  try{ if (username) sessionMemory?.usernames?.add?.(username); if (userid) sessionMemory?.userids?.add?.(String(userid)); }catch{};

      updateStatus(`✅ 已保存: ${manualStatus}`, 'success');
      try {
          const searchTerm = document.getElementById('xhs-search-term')?.value.trim() || "";
          const email = document.getElementById('xhs-email')?.value.trim() || "";
          
          const { username, userid } = grabData();
          if (username || userid) safeSendMessage({ action: 'user_reviewed', user: { username, userid } });
          
          const payload = { username, userid, ...getStats(), url: window.location.href.split('?')[0], email, search_term: searchTerm, status: manualStatus, bio: scrapeBio(), notes: scrapeNotesByTemplate() };
          postJSON(FLASK_MARK_URL, payload).catch(err => { console.error(`❌ 后台发送保存请求失败: ${err}`); });
      } catch (e) { console.error(`❌ 准备数据时出错: ${e}`); }
      if (closePage) {
          safeSendMessage({ action: 'task_completed' });
          setTimeout(requestCloseTab, 200);
      }
  }
  async function renderAiSuggestion() { const row = document.getElementById("ai-suggest-row"); if (!isProfilePage() || !row) return; if (new URLSearchParams(window.location.search).has('rereview')) { row.textContent = "人工复审模式，AI已禁用。"; row.style.color = "#ffc107"; return; } if (globalAiEnabled) { row.textContent = "AI决策模式已开启"; row.style.color = "#ffdd99"; return; } row.textContent = "AI建议：读取中…"; const payload = { ...grabData(), ...getStats(), bio: scrapeBio(), notes: scrapeNotesByTemplate() }; const data = await postJSON(FLASK_AI_SUGGEST_URL, payload); if (data && data.decision && data.decision !== 'error') { const decisionText = { '符合': '符合', '不符合': '不符合', '人工审核': '需人工审核' }[data.decision] || data.decision; const decisionColor = { '符合': '#a6ffa6', '不符合': '#ffc107', '人工审核': '#ffdd99' }[data.decision] || '#ff7676'; row.textContent = `AI建议：${decisionText} (P=${data.p_base?.toFixed(3)})`; row.style.color = decisionColor; } else { row.textContent = `AI建议：获取失败 (${data?.reason || '无响应'})`; row.style.color = "#ff7676"; } }
  function updatePanelHints() { const hintEl = document.getElementById('xhs-mode-hint'); if (hintEl) { hintEl.textContent = globalAiEnabled ? "AI决策模式将连续审核" : "AI建议模式将单次审核"; } }

  function injectStyles() { if (document.getElementById('xhs-panel-styles')) return; const styleSheet = document.createElement("style"); styleSheet.id = 'xhs-panel-styles'; styleSheet.textContent = ` #xhs-review-panel .profile-only, #xhs-review-panel .search-only, #xhs-review-panel .search-only-block, #xhs-review-panel .search-only-flex { display: none; } #xhs-review-panel.is-profile-page .profile-only { display: block; } #xhs-review-panel.is-search-page .search-only { display: flex; } #xhs-review-panel.is-search-page .search-only-block { display: block; } #xhs-review-panel.is-search-page .search-only-flex { display: flex; } `; document.head.appendChild(styleSheet); }

  function createReviewPanel() {
      if (document.getElementById("xhs-review-panel")) return;
      injectStyles();
      const panel = document.createElement('div');
      panel.id = 'xhs-review-panel';
      panel.className = 'unknown-page';
      panel.style.cssText = `position:fixed; top:20px; right:20px; background:#222; border:2px solid #ccc; padding:15px; border-radius:8px; z-index:2147483646; width:300px; font-family:sans-serif; color:#fff;`;
      const inputStyle = `width:100%;box-sizing:border-box;padding:5px;border:1px solid #555;background:#111;color:#eee;border-radius:4px;`;
      const btnStyle = `flex:1;border:none;padding:8px;border-radius:4px;cursor:pointer;`;
      panel.innerHTML = ` <h3 style="margin-top:0;font-size:16px;">审核面板 v10.21</h3> <div class="search-only" style="margin-bottom:8px;gap:4px;"><button id="xhs-start-stop-btn" style="${btnStyle}background:#4CAF50;color:#fff;">开始审核</button><button id="xhs-export-btn" style="${btnStyle}background:#2196F3;color:#fff;">增量同步</button></div> <div id="xhs-mode-hint" class="search-only-block" style="font-size:12px;color:#bbb;margin-bottom:12px;"></div> <hr class="search-only-block" style="border-color:#444; margin: 10px 0;"> <div class="search-only-flex" style="justify-content: space-between; align-items: center; margin-bottom: 8px;"><label style="display:flex;align-items:center;gap:6px;cursor:pointer;"><input type="checkbox" id="ai_auto_toggle"/> <span id="ai-mode-label">AI 模式</span></label><div><button id="rereview_btn" style="padding:4px 8px;border-radius:6px;border:1px solid #ffc107;background:transparent;color:#ffc107;cursor:pointer;margin-right:5px;">复审</button><button id="dashboard_btn" style="padding:4px 8px;border-radius:6px;border:1px solid #61dafb;background:transparent;color:#61dafb;cursor:pointer;">面板</button></div></div> <div class="search-only-block"><label style="display:flex;align-items:center;gap:6px;cursor:pointer;"><input type="checkbox" id="save_history_toggle"/> <span>保存“不符合”数据</span></label></div> <div class="profile-only"><div style="margin-bottom:10px;"><label>搜索关键词:</label><input type="text" id="xhs-search-term" style="${inputStyle}"></div><div style="margin-bottom:10px;"><label>邮箱号:</label><input type="text" id="xhs-email" style="${inputStyle}"></div><div style="display:flex; gap:10px; margin-top: 15px;"><button id="xhs-approve-btn" style="flex:1; padding:10px; background:#28a745; color:white; border:none; border-radius:5px; cursor:pointer; font-size:16px;">✅ 符合</button><button id="xhs-reject-btn" style="flex:1; padding:10px; background:#dc3545; color:white; border:none; border-radius:5px; cursor:pointer; font-size:16px;">❌ 不符合</button></div><div id="ai-suggest-row" style="margin-top:10px;font-size:12px;min-height:1em;"></div></div> <div id="xhs-status" style="margin-top:12px;font-size:12px;color:#ccc;min-height:1em;">正在侦测页面...</div> `;
      document.body.appendChild(panel);
  }

  async function runAutomatedChecks() {
    if (new URLSearchParams(window.location.search).has('rereview')) { updateStatus('人工复审模式，自动化已禁用', 'warn'); renderAiSuggestion(); return; }
    
    updateStatus('正在执行自动化检查...', 'info');
    await sleep(1000);
    const [names, ids] = await Promise.all([GET_JSON(FLASK_URL_USERNAMES), GET_JSON(FLASK_URL_USERIDS)]);
    let knownUsernames = new Set(names || []);
    let knownUserids = new Set(ids || []);

    const { username, userid } = grabData();
    if (!username && !userid) { updateStatus('无法获取用户信息，等待人工', 'warn'); renderAiSuggestion(); return; }

    if ((username && knownUsernames.has(username)) || (userid && knownUserids.has(userid))) {await postJSON((window.FLASK_BASE || FLASK_BASE || '') + '/touch_user', {
  username,
  userid,
  url: window.location.href.split('?')[0]
});

        updateStatus('⚠️ 已在库中, 自动关闭', 'warn');
        safeSendMessage({ action: 'user_reviewed', user: { username, userid } });
        safeSendMessage({ action: 'task_completed' }); 
        setTimeout(requestCloseTab, 1200);
        return;
    }
    const { followers } = getStats();
    if (followers < 100) {
        updateStatus(`⚠️ 粉丝(${followers})<100, 自动标记并关闭`, 'warn');
        await markData('不符合', true);
        return;
    }
    if (globalAiEnabled) {
        updateStatus('🤖 硬规则通过, 转交AI决策...', 'info');
        const searchTerm = document.getElementById('xhs-search-term')?.value.trim() || "";
        const payload = { ...grabData(), ...getStats(), url: window.location.href.split('?')[0], email: "", search_term: searchTerm, bio: scrapeBio(), notes: scrapeNotesByTemplate() };
        const aiResult = await postJSON(FLASK_AI_DECIDE_URL, payload);
        
        if (aiResult?.decision === '符合' || aiResult?.decision === '不符合' || aiResult?.decision === '人工审核') {
            const finalStatus = aiResult.decision;
            updateStatus(`🤖 AI决策: ${finalStatus} (P=${aiResult.p_base?.toFixed(3)})`, 'success');
            safeSendMessage({ action: 'task_completed' }); 
            setTimeout(requestCloseTab, 200);
        } else {
            updateStatus(`❌ AI决策失败: ${aiResult?.reason || '无响应'}. 等待人工`, 'error');
        }
    } else {
        updateStatus('⏳ 硬规则通过, 等待人工审核', 'info');
        renderAiSuggestion();
    }
  }

  function bindPanelEvents() {
      if (document.body.dataset.xhsEventsBound) return;
      document.body.dataset.xhsEventsBound = 'true';

      document.getElementById('xhs-start-stop-btn')?.addEventListener('click', mainAuditLoop);
      document.getElementById('xhs-export-btn')?.addEventListener('click', () => window.open(FLASK_EXPORT_URL, '_blank'));
      document.getElementById('dashboard_btn')?.addEventListener('click', () => window.open(FLASK_DASHBOARD_URL, '_blank'));
      document.getElementById('rereview_btn')?.addEventListener('click', async () => { if (auditTaskState.isRunning) { alert("请先停止当前审核任务！"); return; } updateStatus("⏳ 正在获取待复审列表...", "info"); const urls = await GET_JSON(FLASK_GET_REVIEW_LIST_URL); if (Array.isArray(urls) && urls.length > 0) { if (!confirm(`找到 ${urls.length} 个待复审的用户，是否立即打开？`)) return; updateStatus(`🚀 正在打开 ${urls.length} 个复审页面...`, "info"); await openTabsSequentially(urls, "", true); updateStatus(`✅ ${urls.length} 个复审页面已打开。`, "success"); } else if (Array.isArray(urls)) { updateStatus("✅ 无待复审的用户。", "success"); alert("当前没有需要复审的用户。"); } else { updateStatus("❌ 获取复审列表失败。", "error"); } });
      
      const aiToggle = document.getElementById("ai_auto_toggle"); 
      if (aiToggle) { aiToggle.checked = globalAiEnabled; aiToggle.addEventListener("change", async (e) => { if (auditTaskState.isRunning) { alert("请先停止审核任务再切换模式！"); e.target.checked = globalAiEnabled; return; } globalAiEnabled = e.target.checked; document.getElementById('ai-mode-label').textContent = `AI ${globalAiEnabled ? '决策' : '建议'}模式`; updatePanelHints(); await safeStorageSet({ ai_enabled: globalAiEnabled }); await postJSON(FLASK_AI_SETTINGS_URL, { enabled: globalAiEnabled }); }); }
      const historyToggle = document.getElementById("save_history_toggle"); 
      if(historyToggle) { historyToggle.checked = globalSaveHistoryEnabled; historyToggle.addEventListener("change", async (e) => { globalSaveHistoryEnabled = e.target.checked; await safeStorageSet({ save_history_enabled: globalSaveHistoryEnabled }); await postJSON(FLASK_SAVE_HISTORY_SETTINGS_URL, { enabled: globalSaveHistoryEnabled }); updateStatus(`保存“不符合”数据已 ${globalSaveHistoryEnabled ? '开启' : '关闭'}`, 'info'); }); }
      
      document.getElementById('xhs-approve-btn')?.addEventListener('click', () => markData('符合', true));
      document.getElementById('xhs-reject-btn')?.addEventListener('click', () => markData('不符合', true));
      
      const panelSearchInput = document.getElementById("xhs-search-term");
      if (panelSearchInput && isProfilePage()) {
          // ========== 【关键词BUG修复】 ==========
          // 个人主页的输入框不再监听输入，因为它只是一个显示器
          panelSearchInput.readOnly = true;
          panelSearchInput.style.background = '#333';
          
          safeLog("🕵️‍♂️ [关键词-调试]", "1. 开始从URL填充关键词输入框...");
          try {
              const urlParams = new URLSearchParams(window.location.search);
              const searchTermFromURL = decodeURIComponent(urlParams.get('search_term') || '');
              safeLog("🕵️‍♂️ [关键词-调试]", `2. 从页面URL中读取到关键词: "${searchTermFromURL}"`);
              panelSearchInput.value = searchTermFromURL;
              safeLog("🕵️‍♂️ [关键词-调试]", `3. 已将关键词 "${searchTermFromURL}" 设置到输入框中。`);
          } catch(e) {
              safeLog("🕵️‍♂️ [关键词-调试]", "2. 从URL读取关键词失败:", e);
          }
      }
  }

  async function runOnPageChange() {
      if (pageDetectorInterval) clearInterval(pageDetectorInterval);
      
      document.body.dataset.xhsEventsBound = '';
      createReviewPanel();
      
      let attempts = 0;
      const maxAttempts = 20;
      
      pageDetectorInterval = setInterval(async () => {
          attempts++;
          const pageNow = isProfilePage() ? 'profile' : (isSearchPage() ? 'search' : 'unknown');
          
          if (pageNow !== 'unknown' || attempts > maxAttempts) {
              clearInterval(pageDetectorInterval);
              pageDetectorInterval = null;
              
              const panel = document.getElementById('xhs-review-panel');
              if(panel) panel.className = pageNow === 'profile' ? 'is-profile-page' : (pageNow === 'search' ? 'is-search-page' : 'unknown-page');

              const data = await safeStorageGet(["ai_enabled", "save_history_enabled"]);
              globalAiEnabled = !!data.ai_enabled;
              globalSaveHistoryEnabled = !!data.save_history_enabled;

              bindPanelEvents();
              updatePanelHints();
              (document.getElementById('ai_auto_toggle')||{}).checked = globalAiEnabled;
              (document.getElementById('save_history_toggle')||{}).checked = globalSaveHistoryEnabled;

              if (pageNow === 'profile') {
                  await runAutomatedChecks();
              } else if (pageNow === 'search') {
                  await syncKnownDataForHighlighting();
                  installMutationObserver();
              }
              if(document.getElementById("xhs-status").textContent === "正在侦测页面..."){
                updateStatus('就绪', 'success');
              }
          }
      }, 250);
  }

  function setupGlobalMessageListener() {
      if (window.xhsListenerAttached) return;
      chrome.runtime.onMessage.addListener((msg) => {
          if (msg.action === 'user_reviewed' && msg.user) {
              // 这里的记忆更新主要用于高亮，审核循环有自己独立的记忆体
              if (msg.user.username) sessionMemory.usernames.add(msg.user.username);
              if (msg.user.userid) sessionMemory.userids.add(msg.user.userid);
              syncKnownDataForHighlighting();
          }
      });
      window.xhsListenerAttached = true;
  }

  function main() {
    setupGlobalMessageListener();
    let lastHref = location.href;
    const observer = new MutationObserver(() => {
        if (location.href !== lastHref) {
            lastHref = location.href;
            runOnPageChange();
        }
    });
    runOnPageChange();
    observer.observe(document.body, { childList: true, subtree: true });
  }

  main();

})();
