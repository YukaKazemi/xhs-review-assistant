document.getElementById("start").addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
            const sections = document.querySelectorAll("#exploreFeeds > section");
            const links = [];
            sections.forEach(sec => {
                const a = sec.querySelector("a[href*='/user/profile/']");
                if (a) links.push(a.href);
            });
            return links.slice(0, 20);
        }
    }, (results) => {
        const links = results[0].result;
        chrome.runtime.sendMessage({ action: "startAudit", links }, (response) => {
            document.getElementById("status").textContent = `已打开 ${response.opened} 个主页`;
        });
    });
});

document.getElementById("export").addEventListener("click", () => {
    chrome.runtime.sendMessage({ action: "exportExcel" }, () => {
        document.getElementById("status").textContent = "正在导出...";
    });
});
