(() => {
  const STORAGE_KEY = 'thesis-tracker-entries-v1';
  const TARGET_KEY = 'thesis-tracker-target-v1';

  const entryForm = document.getElementById('entryForm');
  const dateInput = document.getElementById('dateInput');
  const pagesInput = document.getElementById('pagesInput');
  const entriesList = document.getElementById('entriesList');
  const totalWrittenEl = document.getElementById('totalWritten');
  const remainingEl = document.getElementById('remaining');
  const progressBar = document.getElementById('progressBar');
  const targetInput = document.getElementById('targetInput');
  const resetBtn = document.getElementById('resetBtn');
  // planner elements
  const planStart = document.getElementById('planStart');
  const planDays = document.getElementById('planDays');
  const skipWeekends = document.getElementById('skipWeekends');
  const generatePlanBtn = document.getElementById('generatePlan');
  const pagesToPlanEl = document.getElementById('pagesToPlan');
  const pagesPerDayEl = document.getElementById('pagesPerDay');
  const planList = document.getElementById('planList');
  const exportCsvBtn = document.getElementById('exportCsv');
  const exportIcsBtn = document.getElementById('exportIcs');

  const PLAN_KEY = 'thesis-tracker-plan-v1';

  let entries = [];

  function save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  }

  function load() {
    const raw = localStorage.getItem(STORAGE_KEY);
    entries = raw ? JSON.parse(raw) : [];
    const t = localStorage.getItem(TARGET_KEY);
    if (t) targetInput.value = parseInt(t, 10);
  }

  function fmtDate(d) {
    try { return new Date(d).toLocaleDateString(); } catch (e) { return d; }
  }

  function render() {
    const target = Math.max(1, parseInt(targetInput.value, 10) || 75);
    const total = entries.reduce((s,e)=>s+Number(e.pages),0);
    const remaining = Math.max(0, target - total);

    totalWrittenEl.textContent = total;
    remainingEl.textContent = remaining;
    progressBar.style.width = Math.min(100, Math.round((total/target)*100)) + '%';

    entriesList.innerHTML = '';
    // sort descending by date
    const sorted = entries.slice().sort((a,b)=> new Date(b.date) - new Date(a.date));
    for (const e of sorted) {
      const li = document.createElement('li');
      const meta = document.createElement('div'); meta.className='meta';
      meta.innerHTML = `<div class="date">${fmtDate(e.date)}</div><div class="pages">${e.pages} pages</div>`;
      const del = document.createElement('button'); del.textContent = 'Delete';
      del.addEventListener('click', ()=>{ removeEntry(e.id); });
      li.appendChild(meta); li.appendChild(del);
      entriesList.appendChild(li);
    }
  }

  function addEntry(date, pages) {
    entries.push({ id: Date.now() + Math.random(), date, pages: Number(pages) });
    save(); render();
  }

  function removeEntry(id) {
    entries = entries.filter(e=>e.id !== id);
    save(); render();
  }

  entryForm.addEventListener('submit', (ev)=>{
    ev.preventDefault();
    const date = dateInput.value || new Date().toISOString().slice(0,10);
    const pages = pagesInput.value;
    if (!pages || Number(pages) < 0) return alert('Enter a valid page count');
    addEntry(date, pages);
    pagesInput.value = '';
    // keep date as is
  });

  targetInput.addEventListener('change', ()=>{
    const t = Math.max(1, parseInt(targetInput.value, 10) || 75);
    targetInput.value = t;
    localStorage.setItem(TARGET_KEY, String(t));
    render();
  });

  resetBtn.addEventListener('click', ()=>{
    if (!confirm('Clear all entries and reset progress?')) return;
    entries = [];
    localStorage.removeItem(STORAGE_KEY);
    render();
  });

  // set default date field to today
  function setToday() {
    const today = new Date().toISOString().slice(0,10);
    dateInput.value = today;
    planStart.value = today;
  }

  // init
  load(); setToday(); render();

  // --- Planner logic ---
  function savePlan(plan) {
    localStorage.setItem(PLAN_KEY, JSON.stringify(plan));
  }

  function loadPlan() {
    const raw = localStorage.getItem(PLAN_KEY);
    return raw ? JSON.parse(raw) : [];
  }

  function isWeekend(d) {
    const day = d.getDay();
    return day === 0 || day === 6;
  }

  function addDaysSkipping(date, add, skipWeekendsFlag) {
    const res = new Date(date);
    let remaining = add;
    while (remaining > 0) {
      res.setDate(res.getDate() + 1);
      if (skipWeekendsFlag && isWeekend(res)) continue;
      remaining--;
    }
    return res;
  }

  function generatePlan(start, days, totalPages, skipWeekendFlag) {
    const plan = [];
    // collect valid dates
    let date = new Date(start);
    for (let i=0; plan.length < days; ) {
      if (skipWeekendFlag && isWeekend(date)) {
        date.setDate(date.getDate() + 1);
        continue;
      }
      plan.push({ date: date.toISOString().slice(0,10), pages: 0, donePages: 0 });
      date.setDate(date.getDate() + 1);
    }

    const base = Math.floor(totalPages / days);
    let remainder = totalPages - base * days;
    for (let i=0;i<plan.length;i++){
      plan[i].pages = base + (remainder>0?1:0);
      plan[i].donePages = 0;
      remainder = Math.max(0, remainder-1);
    }
    return plan;
  }

  function renderPlan(plan) {
    planList.innerHTML = '';
    for (const p of plan) {
      const li = document.createElement('li');
      const left = document.createElement('div'); left.className='left';
      const dateDiv = document.createElement('div'); dateDiv.className='date'; dateDiv.textContent = new Date(p.date).toLocaleDateString();
      const input = document.createElement('input'); input.className='planned-input'; input.type='number'; input.min=0; input.value = p.pages;
      input.addEventListener('change', ()=>{
        p.pages = Math.max(0, parseInt(input.value,10)||0);
        savePlan(plan);
        updatePlanSummary(plan);
        renderPlan(plan);
      });
      left.appendChild(dateDiv); left.appendChild(input);

      const right = document.createElement('div');
      const status = document.createElement('div'); status.className='status';
      const done = Number(p.donePages||0);
      const planned = Number(p.pages||0);
      let statusText = `${done} / ${planned} pages`;
      status.textContent = statusText;
      if (done >= planned && planned > 0) status.classList.add('done');
      if (done > planned) status.classList.add('over');

      const markBtn = document.createElement('button'); markBtn.textContent = 'Mark done';
      markBtn.addEventListener('click', ()=>{
        p.donePages = p.pages;
        savePlan(plan); renderPlan(plan); updatePlanSummary(plan);
      });

      const resetBtn = document.createElement('button'); resetBtn.textContent = 'Reset done';
      resetBtn.addEventListener('click', ()=>{
        p.donePages = 0; savePlan(plan); renderPlan(plan); updatePlanSummary(plan);
      });

      right.appendChild(status); right.appendChild(markBtn); right.appendChild(resetBtn);
      li.appendChild(left); li.appendChild(right);
      planList.appendChild(li);
    }
  }

  function updatePlanSummary(plan) {
    const total = plan.reduce((s,p)=>s+Number(p.pages),0);
    pagesToPlanEl.textContent = total;
    pagesPerDayEl.textContent = plan.length ? (total/plan.length).toFixed(1) : '0';
  }

  generatePlanBtn.addEventListener('click', ()=>{
    const days = Math.max(1, parseInt(planDays.value,10)||15);
    const start = planStart.value || new Date().toISOString().slice(0,10);
    const skip = Boolean(skipWeekends.checked);
    // pages to plan: remaining pages toward target
    const target = Math.max(1, parseInt(targetInput.value,10)||75);
    const totalWritten = entries.reduce((s,e)=>s+Number(e.pages),0);
    const remaining = Math.max(0, target - totalWritten);
    if (remaining <= 0) {
      alert('You have already reached your target.');
      return;
    }
    const plan = generatePlan(start, days, remaining, skip);
    savePlan(plan);
    renderPlan(plan);
    updatePlanSummary(plan);
    // auto-sync with existing entries so donePages reflects current data
    syncEntriesToPlan(plan);
  });

  // load existing plan on startup
  const existingPlan = loadPlan();
  if (existingPlan && existingPlan.length) { renderPlan(existingPlan); updatePlanSummary(existingPlan); }

  // sync plan with entries
  function syncEntriesToPlan(plan){
    const byDate = {};
    for (const e of entries) {
      byDate[e.date] = (byDate[e.date]||0)+Number(e.pages);
    }
    for (const p of plan) {
      p.donePages = byDate[p.date] || 0;
    }
    savePlan(plan);
    renderPlan(plan);
    updatePlanSummary(plan);
  }

  // sync wrapper that loads stored plan
  function syncStoredPlanWithEntries(){
    const plan = loadPlan();
    if (!plan || !plan.length) return;
    syncEntriesToPlan(plan);
  }

  // call sync on startup
  syncStoredPlanWithEntries();

  // call sync when entries change: after add/remove
  const oldAddEntry = addEntry;
  addEntry = function(date,pages){ oldAddEntry(date,pages); syncStoredPlanWithEntries(); };

  const oldRemove = removeEntry;
  removeEntry = function(id){ oldRemove(id); syncStoredPlanWithEntries(); };

  function download(filename, content, mime='text/csv'){
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }

  exportCsvBtn.addEventListener('click', ()=>{
    const plan = loadPlan();
    if (!plan.length) return alert('No plan available. Generate one first.');
    let csv = 'date,planned,done\n';
    for (const p of plan) csv += `${p.date},${p.pages},${p.donePages||0}\n`;
    download('thesis-plan.csv', csv, 'text/csv');
  });

  exportIcsBtn.addEventListener('click', ()=>{
    const plan = loadPlan();
    if (!plan.length) return alert('No plan available. Generate one first.');
    const uidBase = Date.now();
    let ics = 'BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//MA Thesis Tracker//EN\n';
    for (let i=0;i<plan.length;i++){
      const p = plan[i];
      const dt = p.date.replace(/-/g,'');
      const uid = `plan-${uidBase}-${i}`;
      const summary = encodeURIComponent(`Write ${p.pages} pages`);
      const desc = `Planned: ${p.pages} pages; Done: ${p.donePages||0}`;
      ics += `BEGIN:VEVENT\nUID:${uid}\nDTSTAMP:${dt}T090000Z\nDTSTART;VALUE=DATE:${dt}\nSUMMARY:Write ${p.pages} pages\nDESCRIPTION:${desc}\nEND:VEVENT\n`;
    }
    ics += 'END:VCALENDAR';
    download('thesis-plan.ics', ics, 'text/calendar');
  });

  // sync button
  const syncPlanBtn = document.getElementById('syncPlan');
  syncPlanBtn.addEventListener('click', ()=>{ syncStoredPlanWithEntries(); alert('Plan synced with entries.'); });

})();
