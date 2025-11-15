// content_ai.js (v10.28 - “等待戈多” Bug修复版)
// - 【核心修复】: 彻底重构了等待批处理完成的机制，解决主页面卡住不动的“等待戈多”问题。
//   1. 废除了在循环中动态“添加/移除”消息监听器的脆弱模式。
//   2. 新增了一个全局的、唯一的 `batchCompletionManager` 状态管理器。
//   3. `setupGlobalMessageListener` 现在会把所有 `task_completed` 消息统一交给 `batchCompletionManager` 处理。
//   4. `waitForBatchBySignal` 函数被重写，它不再操作监听器，而是向 `batchCompletionManager` 发起一个“等待契约”，返回一个由管理器在未来解决的 Promise。
// - 【健壮性】: 这种新的中心化事件处理架构，完全避免了事件监听器在循环中的竞态和丢失问题，确保审核流程可以稳定地、连续地进行。
// - 【超时调整】: 略微增加了批处理的超时时间，以应对网络波动。

(function () {
  'use strict';

  console.log('[DEBUG-JS-LOAD] content_ai.js v10.28 (“等待戈多” Bug修复版) 脚本开始执行...');

  const FLASK_BASE = "http://localhost:5001";
  const FLASK_MARK_URL = `${FLASK_BASE}/mark_data`;
  const FLASK_URL_USERNAMES = `${FLASK_BASE}/usernames`;
  const FLASK_URL_USERIDS = `${FLASK_BASE}/userids`;
  const FLASK_AI_DECIDE_URL = `${FLASK_BASE}/ai/decide`;
  const FLASK_AI_SUGGEST_URL = `${FLASK_BASE}/ai/suggest`;
  const FLASK_AI_SETTINGS_URL = `${FLASK_BASE}/ai/settings`;
  const FLASK_SAVE_HISTORY_SETTINGS_URL = `${FLASK_BASE}/settings/save_history`;
  const FLASK_EXPORT_URL = `${FLASK_BASE}/export_delta?dataset=approved`;
  const FLASK_GET_REVIEW_LIST_URL = `${FLASK_BASE}/get_review_list`;
  const FLASK_DASHBOARD_URL = `${FLASK_BASE}/dashboard`;
  const FLASK_TOUCH_USER_URL = `${FLASK_BASE}/touch_user`;

  const SEARCH_TERM_KEY = "xhs_last_search_term";

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

  const sessionMemory = {
      usernames: new Set(),
      userids: new Set()
  };
  
  // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 【核心修复 I】 新增全局状态管理器 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
  const batchCompletionManager = {
      waiter: null,
      completedCount: 0,
      expectedCount: 0,
      timeoutHandle: null,
  
      signalCompletion: function() {
          if (!this.waiter) return;
          this.completedCount++;
          updateStatus(`⏳ 等待 ${this.expectedCount} 个页面完成... (${this.completedCount}/${this.expectedCount})`, "info");
          if (this.completedCount >= this.expectedCount) {
              this.resolveWaiter(true, "✅ 当前批次已完成。");
          }
      },
      
      resolveWaiter: function(result, message) {
          if (!this.waiter) return;
          if (this.timeoutHandle) {
              clearTimeout(this.timeoutHandle);
              this.timeoutHandle = null;
          }
          updateStatus(message, result ? "success" : "warn");
          const resolver = this.waiter;
          this.waiter = null;
          resolver(result);
      },
  
      startWaiting: function(batchSize) {
          if (batchSize === 0) return Promise.resolve(true);
          return new Promise(resolve => {
              this.waiter = resolve;
              this.completedCount = 0;
              this.expectedCount = batchSize;
              updateStatus(`⏳ 等待 ${batchSize} 个页面完成... (0/${batchSize})`, "info");
              this.timeoutHandle = setTimeout(() => {
                  this.resolveWaiter(true, "⚠️ 等待超时，可能部分页面未响应");
              }, batchSize * 20 * 1000);
          });
      }
  };
  // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 【核心修复 I】 新增全局状态管理器 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

  const sleep = (ms) => new Promise(r=>setTimeout(r, ms));
  function extAlive(){ try { return !!(chrome && chrome.runtime && chrome.runtime.id); } catch { return false; } }
  function safeSendMessage(msg){
      console.log(`[DEBUG-JS-MSG] -> Sending message to background:`, msg);
      return new Promise((resolve)=> {
        if (!extAlive()) return resolve({ ok:false, err:"invalidated" });
        try {
          chrome.runtime.sendMessage(msg, (res)=>{
            const err = chrome.runtime.lastError;
            if (err) { console.error(err); return resolve({ ok:false, err:err.message }); }
            resolve(res || { ok:true });
          });
        } catch(e){
          console.error(e);
          resolve({ ok:false, err:String(e) });
        }
      });
  }
  function safeStorageGet(keys){ return new Promise((resolve)=>{ if (!extAlive()) return resolve({}); try { chrome.storage.local.get(keys, resolve); } catch { resolve({}); } }); }
  function safeStorageSet(obj){ return new Promise((resolve)=>{ if (!extAlive()) return resolve(false); try { chrome.storage.local.set(obj, ()=> resolve(!chrome.runtime.lastError)); } catch { resolve(false); } }); }
  function requestCloseTab(){ safeSendMessage({ action:"closeTab", url: window.location.href }); }
  function safeLog(prefix, ...args){ try{ console.log(prefix, ...args);}catch(e){} }

  // --- 改用后台代理请求，以避免 HTTPS 页面混合内容问题 ---
  async function GET_JSON(url) {
      try {
          const r = await safeSendMessage({ action: 'bg_fetch_json', url });
          return (r && r.ok) ? r.data : null;
      } catch (e) {
          console.error('GET_JSON via bg failed:', e);
          return null;
      }
  }

  async function postJSON(url, data) {
      try {
          const r = await safeSendMessage({ action: 'bg_post_json', url, data });
          return (r && r.ok) ? r.data : null;
      } catch (e) {
          console.error('postJSON via bg failed:', e);
          return null;
      }
  }
  
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
  
  async function syncKnownDataForHighlighting() {
    const [names, ids] = await Promise.all([GET_JSON(FLASK_URL_USERNAMES), GET_JSON(FLASK_URL_USERIDS)]);
    let knownUsernames = new Set(Array.isArray(names) ? names : []);
    let knownUserids = new Set(Array.isArray(ids) ? ids : []);
    document.querySelectorAll("a[href*='/user/profile/']").forEach(el => {
        const name = el.querySelector(".name, .user-name")?.textContent.trim();
        const uid = el.getAttribute("data-user-id")?.trim() || el.closest("[data-user-id]")?.getAttribute("data-user-id")?.trim();
        if ((name && knownUsernames.has(name)) || (uid && knownUserids.has(uid))) {
            highlightElement(el.closest("figure, .note-item, section, .user-item"));
        }
    });
  }

  function installMutationObserver(){ const containers = [ document.querySelector("#exploreFeeds"), document.querySelector(".search-layout__main .feeds-container"), document.querySelector("#global .search-layout__main .feeds-container"), document.querySelector(".user-list"), document.querySelector("#user-list"), document.querySelector(".feeds-container") ].filter(Boolean); if (!containers.length) return; const ob = new MutationObserver(() => { clearTimeout(ob._tid); ob._tid = setTimeout(syncKnownDataForHighlighting, 120); }); containers.forEach(c => ob.observe(c, { childList: true, subtree: true })); }
  
  async function openTabsSequentially(urls, isRereview = false){ 
      const uniqUrls=Array.from(new Set(urls.filter(Boolean)));
      let opened = 0;
      let idx = 0;
      for (const url of uniqUrls) {
          if (auditTaskState.shouldStop) break;
          idx++;
          let finalUrl = url;
          if (isRereview) {
              try {
                  const urlObj = new URL(url.startsWith('http') ? url : `https://www.xiaohongshu.com${url}`);
                  urlObj.searchParams.set('rereview', '1');
                  finalUrl = urlObj.toString();
              } catch(e) {/* plain url is fine */}
          }
          const res = await safeSendMessage({ action:"openTab", url: finalUrl });
          // ⭐ 关键修复：只有明确 {ok:true} 才计入“成功打开”
          if (res && res.ok === true) {
              opened++;
          } else {
              console.warn("[openTabsSequentially] failed to open:", finalUrl, "res=", res);
          }
          await sleep(Math.random() * (HUMAN_MAX_DELAY_MS - HUMAN_MIN_DELAY_MS) + HUMAN_MIN_DELAY_MS);
      }
      return opened;
  }
  
  function setButtonState(isAuditing) { const startStopBtn = document.getElementById('xhs-start-stop-btn'); if (startStopBtn) { startStopBtn.textContent = isAuditing ? "停止审核" : "开始审核"; startStopBtn.style.background = isAuditing ? "#dc3545" : "#4CAF50"; } }
  function findScrollContainer() { const candidates = [document.scrollingElement, document.documentElement, document.body, document.querySelector('.main-content'), document.querySelector('#app-container')]; for (const el of candidates) { if (el && el.scrollHeight > el.clientHeight + 100) return el; } return document.scrollingElement || document.documentElement; }
  async function autoScrollAndHarvestAll(){ const container=findScrollContainer(); const found=new Map(); let lastSeen=0; let idleRounds=0; for (let round=0; round<500; round++){ if(auditTaskState.shouldStop) break; snapshotCardsInto(found); container.scrollTop += AUTOSCROLL_STEP; await sleep(AUTOSCROLL_ROUND_WAIT); const curCount=found.size; idleRounds=(curCount<=lastSeen)?(idleRounds+1):0; lastSeen=curCount; if (idleRounds >= AUTOSCROLL_IDLE_ROUNDS) break; } container.scrollTop = 0; return Array.from(found.values()); }
  function snapshotCardsInto(map){ document.querySelectorAll("a[href*='/user/profile/']").forEach(link => { const href = (link.href || "").trim(); if (!href) return; const nameElem = link.querySelector(".name, .user-name, .user-nickname, .author-name, .nickname"); const uname = (nameElem?.textContent || "").trim(); const userItem = link.closest('.user-item, section, figure, .note-item, .item, [data-user-id]'); const uid = userItem?.getAttribute("data-user-id") || ""; const key = uid || href; if (!map.has(key)) { map.set(key, { href, uname, uid }); } }); }
  function filterNotReviewed(cards){ return cards.filter(({uname, uid})=>{ const u = (uname || "").trim(); const i = (uid || "").trim(); if (!u && !i) return false; return !(sessionMemory.usernames.has(u) || sessionMemory.userids.has(i)); }); }
  
  // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 【核心修复 II】 重写等待函数 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
  function waitForBatchBySignal(batchSize) {
      return batchCompletionManager.startWaiting(batchSize);
  }
  // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 【核心修复 II】 重写等待函数 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

  async function mainAuditLoop() {
      if (auditTaskState.isRunning) { auditTaskState.shouldStop = true; setButtonState(false); updateStatus("⛔ 正在停止任务...", "warn"); return; }
      Object.assign(auditTaskState, { isRunning: true, shouldStop: false });
      setButtonState(true);
      try {
          updateStatus("⏬ 同步“长期记忆”到会话...", "info");
          const [names, ids] = await Promise.all([GET_JSON(FLASK_URL_USERNAMES), GET_JSON(FLASK_URL_USERIDS)]);
          if (names === null || ids === null) throw new Error("后端连接失败，任务中止。");
          sessionMemory.usernames = new Set(names);
          sessionMemory.userids = new Set(ids);
          let batchNum = 1;
          while (!auditTaskState.shouldStop) {
              updateStatus(`⏬ 自动滚动，采集第 ${batchNum} 批新用户...`, "info");
              const allCards = await autoScrollAndHarvestAll();
              const candidates = filterNotReviewed(allCards);
              if (candidates.length === 0) { updateStatus("✅ 所有可见用户均已审核完毕！", "success"); break; }
              const batchToOpen = candidates.slice(0, OPEN_BATCH_LIMIT);
              for (const user of batchToOpen) {
                  if (user.uname) sessionMemory.usernames.add(user.uname);
                  if (user.uid) sessionMemory.userids.add(user.uid);
              }
              
              const searchInputs = document.querySelectorAll('input[placeholder*="搜索"], #search-input');
              let currentSearchTerm = "";
              if (searchInputs.length > 0) {
                  for (const input of searchInputs) {
                      if (input.value && input.offsetParent !== null) {
                          currentSearchTerm = input.value.trim();
                          break;
                      }
                  }
              }
              if (currentSearchTerm) {
                  console.log(`[DEBUG-JS-KEYWORD] 捕获到搜索词: "${currentSearchTerm}", 正在发送给后台存储...`);
                  await safeSendMessage({ action: "save_search_term", term: currentSearchTerm });
              }

              updateStatus(`🚀 正在打开第 ${batchNum} 批 (共 ${batchToOpen.length} 个)...`, "info");
              const openedCount = await openTabsSequentially(batchToOpen.map(c => c.href));
              
              if (auditTaskState.shouldStop) break;
              if (globalAiEnabled) {
                  await waitForBatchBySignal(openedCount);
                  if (auditTaskState.shouldStop) break;
                  batchNum++;
                  await sleep(1000);
              } else {
                  updateStatus(`✅ 已打开 ${openedCount} 个页面供您审核。`, "success");
                  break;
              }
          }
      } catch (e) {
          updateStatus(`❌ 审核循环出错: ${e.message}`, "error");
          console.error(e);
      } finally {
          if (auditTaskState.shouldStop) updateStatus("⛔ 任务已手动停止。", "warn");
          Object.assign(auditTaskState, { isRunning: false, shouldStop: false });
          setButtonState(false);
      }
  }

  async function markData(manualStatus, closePage = true) {
      try{ const {username, userid} = grabData(); if (username) sessionMemory?.usernames?.add?.(username); if (userid) sessionMemory?.userids?.add?.(String(userid)); }catch{};
      updateStatus(`✅ 已保存: ${manualStatus}`, 'success');
      try {
          const searchTerm = document.getElementById('xhs-search-term')?.value.trim() || "";
          const email = document.getElementById('xhs-email')?.value.trim() || "";
          const { username, userid } = grabData();
          if (username || userid) safeSendMessage({ action: 'user_reviewed', user: { username, userid } });
          const isRereview = new URLSearchParams(window.location.search).has('rereview');
          const payload = { 
              username, 
              userid, 
              ...getStats(), 
              url: window.location.href.split('?')[0], 
              email, 
              search_term: searchTerm, 
              status: manualStatus, 
              bio: scrapeBio(), 
              notes: scrapeNotesByTemplate(),
              is_rereview: isRereview
          };
          postJSON(FLASK_MARK_URL, payload).catch(err => console.error(`❌ 后台发送保存请求失败: ${err}`));
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
      panel.innerHTML = ` <h3 style="margin-top:0;font-size:16px;">审核面板 v10.28</h3> <div class="search-only" style="margin-bottom:8px;gap:4px;"><button id="xhs-start-stop-btn" style="${btnStyle}background:#4CAF50;color:#fff;">开始审核</button><button id="xhs-export-btn" style="${btnStyle}background:#2196F3;color:#fff;">增量同步</button></div> <div id="xhs-mode-hint" class="search-only-block" style="font-size:12px;color:#bbb;margin-bottom:12px;"></div> <hr class="search-only-block" style="border-color:#444; margin: 10px 0;"> <div class="search-only-flex" style="justify-content: space-between; align-items: center; margin-bottom: 8px;"><label style="display:flex;align-items:center;gap:6px;cursor:pointer;"><input type="checkbox" id="ai_auto_toggle"/> <span id="ai-mode-label">AI 模式</span></label><div><button id="rereview_btn" style="padding:4px 8px;border-radius:6px;border:1px solid #ffc107;background:transparent;color:#ffc107;cursor:pointer;margin-right:5px;">复审</button><button id="dashboard_btn" style="padding:4px 8px;border-radius:6px;border:1px solid #61dafb;background:transparent;color:#61dafb;cursor:pointer;">面板</button></div></div> <div class="search-only-block"><label style="display:flex;align-items:center;gap:6px;cursor:pointer;"><input type="checkbox" id="save_history_toggle"/> <span>保存“不符合”数据</span></label></div> <div class="profile-only"><div style="margin-bottom:10px;"><label>搜索关键词:</label><input type="text" id="xhs-search-term" style="${inputStyle}"></div><div style="margin-bottom:10px;"><label>邮箱号:</label><input type="text" id="xhs-email" style="${inputStyle}"></div><div style="display:flex; gap:10px; margin-top: 15px;"><button id="xhs-approve-btn" style="flex:1; padding:10px; background:#28a745; color:white; border:none; border-radius:5px; cursor:pointer; font-size:16px;">✅ 符合</button><button id="xhs-reject-btn" style="flex:1; padding:10px; background:#dc3545; color:white; border:none; border-radius:5px; cursor:pointer; font-size:16px;">❌ 不符合</button></div><div id="ai-suggest-row" style="margin-top:10px;font-size:12px;min-height:1em;"></div></div> <div id="xhs-status" style="margin-top:12px;font-size:12px;color:#ccc;min-height:1em;">正在侦测页面...</div> `;
      document.body.appendChild(panel);
  }
  async function runAutomatedChecks() {
    if (new URLSearchParams(window.location.search).has('rereview')) { updateStatus('人工复审模式，自动化已禁用', 'warn'); renderAiSuggestion(); return; }
    updateStatus('正在执行自动化检查...', 'info'); await sleep(1000);
    const { username, userid } = grabData();
    if (!username && !userid) { updateStatus('无法获取用户信息，等待人工', 'warn'); renderAiSuggestion(); return; }
    const [names, ids] = await Promise.all([ GET_JSON(FLASK_URL_USERNAMES), GET_JSON(FLASK_URL_USERIDS) ]);
    if (names === null || ids === null) { updateStatus('❌ 后端连接失败，自动化中止', 'error'); return; }
    const knownUsernames = new Set(names); const knownUserids = new Set(ids);
    const isUsernameKnown = username && knownUsernames.has(username); const isUseridKnown = userid && knownUserids.has(String(userid));
    if (isUsernameKnown || isUseridKnown) {
        updateStatus('⚠️ 已在库中, 正在补充记录并关闭...', 'warn');
        postJSON(FLASK_TOUCH_USER_URL, { username, userid, url: window.location.href.split('?')[0] });
        safeSendMessage({ action: 'user_reviewed', user: { username, userid } });
        safeSendMessage({ action: 'task_completed' }); 
        setTimeout(requestCloseTab, 1200);
        return; 
    }
    const { followers } = getStats();
    if (followers < 100) { updateStatus(`⚠️ 粉丝(${followers})<100, 自动标记并关闭`, 'warn'); await markData('不符合', true); return; }
    if (globalAiEnabled) {
        updateStatus('🤖 硬规则通过, 转交AI决策...', 'info');
        const searchTerm = document.getElementById('xhs-search-term')?.value.trim() || "";
        const payload = { ...grabData(), ...getStats(), url: window.location.href.split('?')[0],
                email: document.getElementById('xhs-email')?.value.trim() || "",
              search_term: searchTerm, bio: scrapeBio(), notes: scrapeNotesByTemplate() };
        const aiResult = await postJSON(FLASK_AI_DECIDE_URL, payload);
        if (aiResult?.decision) {
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
      document.getElementById('rereview_btn')?.addEventListener('click', async () => { if (auditTaskState.isRunning) { alert("请先停止当前审核任务！"); return; } updateStatus("⏳ 正在获取待复审列表...", "info"); const urls = await GET_JSON(FLASK_GET_REVIEW_LIST_URL); if (Array.isArray(urls) && urls.length > 0) { if (!confirm(`找到 ${urls.length} 个待复审的用户，是否立即打开？`)) return; updateStatus(`🚀 正在打开 ${urls.length} 个复审页面...`, "info"); await openTabsSequentially(urls, true); updateStatus(`✅ ${urls.length} 个复审页面已打开。`, "success"); } else if (Array.isArray(urls)) { updateStatus("✅ 无待复审的用户。", "success"); alert("当前没有需要复审的用户。"); } else { updateStatus("❌ 获取复审列表失败。", "error"); } });
      const aiToggle = document.getElementById("ai_auto_toggle"); 
      if (aiToggle) { aiToggle.checked = globalAiEnabled; aiToggle.addEventListener("change", async (e) => { if (auditTaskState.isRunning) { alert("请先停止审核任务再切换模式！"); e.target.checked = globalAiEnabled; return; } globalAiEnabled = e.target.checked; document.getElementById('ai-mode-label').textContent = `AI ${globalAiEnabled ? '决策' : '建议'}模式`; updatePanelHints(); await safeStorageSet({ ai_enabled: globalAiEnabled }); await postJSON(FLASK_AI_SETTINGS_URL, { enabled: globalAiEnabled }); }); }
      const historyToggle = document.getElementById("save_history_toggle"); 
      if(historyToggle) { historyToggle.checked = globalSaveHistoryEnabled; historyToggle.addEventListener("change", async (e) => { globalSaveHistoryEnabled = e.target.checked; await safeStorageSet({ save_history_enabled: globalSaveHistoryEnabled }); await postJSON(FLASK_SAVE_HISTORY_SETTINGS_URL, { enabled: globalSaveHistoryEnabled }); updateStatus(`保存“不符合”数据已 ${globalSaveHistoryEnabled ? '开启' : '关闭'}`, 'info'); }); }
      document.getElementById('xhs-approve-btn')?.addEventListener('click', () => markData('符合', true));
      document.getElementById('xhs-reject-btn')?.addEventListener('click', () => markData('不符合', true));
      const panelSearchInput = document.getElementById("xhs-search-term");
      if (panelSearchInput && isProfilePage()) {
          panelSearchInput.readOnly = true;
          panelSearchInput.style.background = '#333';
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
                  const storageData = await safeStorageGet([SEARCH_TERM_KEY]);
                  const term = storageData[SEARCH_TERM_KEY];
                  if (term) {
                      const panelInput = document.getElementById("xhs-search-term");
                      if (panelInput) { panelInput.value = term; }
                  }
                  await runAutomatedChecks();
              } else if (pageNow === 'search') {
                  await syncKnownDataForHighlighting();
                  installMutationObserver();
                  try {
                      const syncKeywordNow = async () => {
                          const inp = document.querySelector('#search-input') || document.querySelector('input[placeholder*="搜索"]');
                          const v = (inp && inp.value || '').trim();
                          if (v) { await safeSendMessage({ action: 'save_search_term', term: v }); }
                      };
                      await syncKeywordNow();
                      const bind = () => {
                          const inp = document.querySelector('#search-input') || document.querySelector('input[placeholder*="搜索"]');
                          if (!inp || inp.__xhs_bind) return;
                          inp.__xhs_bind = true;
                          const handler = async () => {
                              const v = (inp.value || '').trim();
                              if (v) { await safeSendMessage({ action: 'save_search_term', term: v }); }
                          };
                          inp.addEventListener('change', handler);
                          inp.addEventListener('blur', handler);
                          inp.addEventListener('input', (() => { let t; return () => { clearTimeout(t); t = setTimeout(handler, 250); }; })());
                      };
                      bind();
                      setTimeout(bind, 1500);
                      setTimeout(bind, 3000);
                  } catch(e) { console.warn('Search page keyword sync failed:', e); }

              }
              if(document.getElementById("xhs-status").textContent === "正在侦测页面..."){ updateStatus('就绪', 'success'); }
          }
      }, 250);
  }
  const debouncedHighlight = debounce(syncKnownDataForHighlighting, 300);

  // ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 【核心修复 III】 改造全局监听器 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
  function setupGlobalMessageListener() {
      if (window.xhsListenerAttached) return;
      chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
          if (msg.action === 'user_reviewed' && msg.user) {
              if (msg.user.username) sessionMemory.usernames.add(msg.user.username);
              if (msg.user.userid) sessionMemory.userids.add(String(msg.user.userid));
              debouncedHighlight();
              sendResponse({status: "ok"});
          } else if (msg.action === 'task_completed') {
              batchCompletionManager.signalCompletion();
          }
      });
      window.xhsListenerAttached = true;
  }
  // ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 【核心修复 III】 改造全局监听器 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

  function main() {
    setupGlobalMessageListener();
    let lastHref = location.href;
    const observer = new MutationObserver(() => {
        if (location.href !== lastHref) { lastHref = location.href; runOnPageChange(); }
    });
    runOnPageChange();
    observer.observe(document.body, { childList: true, subtree: true });
  }

  main();
})();


// === Added by patch: storage change listener to auto-hydrate profile panel ===
try {
  if (!window.__xhs_storage_listener_bound) {
    window.__xhs_storage_listener_bound = true;
    chrome.storage.onChanged.addListener((changes, area) => {
      const SEARCH_TERM_KEY = "xhs_last_search_term";
      if (area === 'local' && changes[SEARCH_TERM_KEY]) {
        const val = (changes[SEARCH_TERM_KEY].newValue || '').trim();
        const input = document.getElementById('xhs-search-term');
        if (input) {
          input.value = val;
        }
      }
    });
  }
} catch (e) { console.warn('Failed to bind storage listener patch:', e); }
