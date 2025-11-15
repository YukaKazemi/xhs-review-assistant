// background.js (v10.28-compatible + robust tab tracking & broadcasts)

const FLASK_BASE = "http://localhost:5001";
const XHS_MATCH = "*://*.xiaohongshu.com/*";
const SEARCH_TERM_KEY = "xhs_last_search_term";

// 记录本扩展“主动创建”的审查标签页：tabId -> { url, ts }
const OPENED_TASK_TABS = new Map();

// 兜底：我们的任务 tab 一旦被关闭，就向所有小红书页广播一次 task_completed
chrome.tabs.onRemoved.addListener((closedId) => {
  if (!OPENED_TASK_TABS.has(closedId)) return;
  const meta = OPENED_TASK_TABS.get(closedId);
  OPENED_TASK_TABS.delete(closedId);

  const msg = { action: "task_completed", reason: "tab_removed_fallback", tabId: closedId, url: meta?.url || "" };
  chrome.tabs.query({}, (tabs) => {
    tabs.forEach(tab => {
      if (tab?.id && tab.url && tab.url.includes("xiaohongshu.com")) {
        try { chrome.tabs.sendMessage(tab.id, msg).catch(()=>{}); } catch (_){}
      }
    });
  });
});

// 工具函数：广播到所有小红书页
async function broadcastToXHS(message) {
  const tabs = await chrome.tabs.query({ url: XHS_MATCH });
  for (const t of tabs) {
    try { await chrome.tabs.sendMessage(t.id, message); } catch (_){}
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // 代理 GET JSON
  if (msg?.action === "bg_fetch_json") {
    (async () => {
      try {
        const res = await fetch(msg.url, { method: "GET" });
        const data = await res.json().catch(()=> ({}));
        sendResponse({ ok: res.ok, status: res.status, data });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    })();
    return true;
  }

  // 代理 POST JSON（兼容 data 或 body）
  if (msg?.action === "bg_post_json") {
    (async () => {
      try {
        const payload = (msg.body !== undefined) ? msg.body : msg.data;
        const res = await fetch(msg.url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: typeof payload === "string" ? payload : JSON.stringify(payload || {})
        });
        const data = await res.json().catch(()=> ({}));
        sendResponse({ ok: res.ok, status: res.status, data });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    })();
    return true;
  }

  // 打开标签
  if (msg?.action === "openTab") {
    (async () => {
      try {
        const url = msg.url;
        if (!url) { sendResponse({ ok:false, error: "missing url" }); return; }
        const tab = await chrome.tabs.create({ url, active: false });
        if (tab && typeof tab.id === "number") {
          OPENED_TASK_TABS.set(tab.id, { url, ts: Date.now() });
          sendResponse({ ok: true, tabId: tab.id });
        } else {
          sendResponse({ ok:false, error: "create tab failed" });
        }
      } catch (e) {
        sendResponse({ ok:false, error: String(e) });
      }
    })();
    return true;
  }

  // 关闭标签
  if (msg?.action === "closeTab") {
    (async () => {
      try {
        const fromId = sender?.tab?.id;
        if (fromId) {
          try { await chrome.tabs.remove(fromId); } catch(_){}
          sendResponse({ ok: true, closedId: fromId });
          return;
        }
        const hint = msg.url || "";
        const tabs = await chrome.tabs.query({ url: XHS_MATCH });
        const target = tabs.find(t => t.url && hint && t.url.split("?")[0] === hint.split("?")[0]);
        if (target?.id) {
          await chrome.tabs.remove(target.id);
          sendResponse({ ok: true, closedId: target.id });
        } else {
          sendResponse({ ok: false, error: "no tab to close" });
        }
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    })();
    return true;
  }

  // 广播（保留）
  if (msg?.action === "broadcast") {
    (async () => {
      try { await broadcastToXHS(msg.payload || {}); sendResponse({ ok: true }); }
      catch (e) { sendResponse({ ok:false, error: String(e) }); }
    })();
    return true;
  }

  // 新增：收到 task_completed / user_reviewed 时，统一广播到所有小红书页
  if (msg?.action === "task_completed" || msg?.action === "user_reviewed") {
    (async () => {
      try { await broadcastToXHS(msg); sendResponse({ ok: true }); }
      catch (e) { sendResponse({ ok:false, error: String(e) }); }
    })();
    return true;
  }

  // 新增：保存搜索词到 storage，让 profile 面板能实时取到
  if (msg?.action === "save_search_term") {
    (async () => {
      try {
        await chrome.storage.local.set({ [SEARCH_TERM_KEY]: String(msg.term || "") });
        sendResponse({ ok: true });
      } catch (e) {
        sendResponse({ ok: false, error: String(e) });
      }
    })();
    return true;
  }

  // （可选）你的其它路由...
});
