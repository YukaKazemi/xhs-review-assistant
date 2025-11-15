// =========================
// background.js (v10.7.3 - 关键词逻辑优先，终极修复版)
// =========================
// 【核心修复】调整了消息处理的逻辑顺序。将 'save_search_term' 的处理逻辑块移至最前，
// 确保它不会被其他消息处理逻辑提前中断。这是解决关键词无法保存的根本原因。
// 【保留】保留所有 v10.7.1 中已验证的设置和功能，不引入任何其他改动。

const SEARCH_TERM_KEY = "xhs_last_search_term";
const FLASK_BASE = "http://localhost:5001";
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  
  // 【核心修复】把关键词处理逻辑放在最前面，优先处理！
  if (msg.action === "save_search_term") {
    const term = (msg.term || "").trim();
    if (!term) { sendResponse({ ok: true, ignored: true, reason: "empty term" }); return; }
    chrome.storage.local.set({ [SEARCH_TERM_KEY]: term }, () => {
      const le = chrome.runtime.lastError;
      if (le) { console.warn('[BG] storage.set error:', le); sendResponse({ ok:false, error:String(le.message||le) }); }
      else { console.log('[BG] term saved:', term); sendResponse({ ok:true }); }
    });
    return true;
  }

  // ---- 以下是您原有的、保持不变的逻辑 ----

  // 广播 'user_reviewed' 和 'task_completed' 信号
  if (msg?.action === 'user_reviewed' || msg?.action === 'task_completed') {
    console.log('Background received broadcast signal:', msg);
    chrome.tabs.query({}, (tabs) => {
      tabs.forEach(tab => {
        if (tab.url && tab.url.includes("xiaohongshu.com")) {
          chrome.tabs.sendMessage(tab.id, msg).catch(err => {
            if (!err.message.includes("Could not establish connection")) {
              console.error('Failed to send message to tab:', tab.id, err);
            }
          });
        }
      });
    });
    // 这个 return false 是正确的，表示广播后就结束了
    return false;
  }

  // 打开标签
  if (msg?.action === "openTab") {
    (async () => {
      const url = msg.url;
      let ok = false, err = null;
      for (let i = 0; i < 4; i++) {
        try {
          await chrome.tabs.create({ url, active: false });
          ok = true; break;
        } catch (e) {
          err = e; await sleep(160 + i * 120);
        }
      }
      sendResponse({ ok, error: err?.message });
    })();
    return true;
  }

  // 关闭标签
  if (msg?.action === "closeTab") {
    (async () => {
      try {
        const fromId = sender?.tab?.id;
        if (fromId) {
          try { await chrome.tabs.remove(fromId); } catch(e){}
          sendResponse({ ok: true }); return;
        }
        const hint = msg.url || "";
        const tabs = await chrome.tabs.query({url: "*://*.xiaohongshu.com/*"});
        const target = tabs.find(t => t.url && hint && t.url.includes(hint.split("?")[0]));
        if (target) {
          await chrome.tabs.remove(target.id);
          sendResponse({ ok: true });
        } else {
          sendResponse({ ok: false, error: "no tab to close" });
        }
      } catch (e) {
        sendResponse({ ok: false, error: e?.message });
      }
    })();
    return true;
  }

  // 同步 Excel
  if (msg?.action === "syncExcel") {
    (async () => {
      try {
        const res = await fetch(`${FLASK_BASE}/sync_excel`, { method: "POST" });
        const json = await res.json();
        sendResponse({ ok: res.ok, ...json });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    })();
    return true;
  }

});
