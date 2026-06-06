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
  // planner elements (manual workflow)
  const planDateInput = document.getElementById('planDate');
  const planPagesInput = document.getElementById('planPages');
  const addPlanBtn = document.getElementById('addPlan');
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
    const rawTarget = parseInt(targetInput.value, 10);
    const target = Number.isFinite(rawTarget) ? rawTarget : null;
    const total = entries.reduce((s,e)=>s+Number(e.pages),0);

    totalWrittenEl.textContent = total;
    if (target) {
      const remaining = Math.max(0, target - total);
      remainingEl.textContent = remaining;
      progressBar.style.width = Math.min(100, Math.round((total/target)*100)) + '%';
    } else {
      remainingEl.textContent = '—';
      progressBar.style.width = '0%';
    }

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

  function addEntry(date, pages, origin, planId) {
    entries.push({ id: Date.now() + Math.random(), date, pages: Number(pages), origin: origin || null, planId: planId || null });
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
    const v = parseInt(targetInput.value, 10);
    if (!Number.isFinite(v) || v < 1) {
      targetInput.value = '';
      localStorage.removeItem(TARGET_KEY);
    } else {
      targetInput.value = v;
      localStorage.setItem(TARGET_KEY, String(v));
    }
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
    if (planDateInput) planDateInput.value = today;
  }

  // init
  load(); setToday(); render();

  // --- Planner logic ---
  function savePlan(plan) {
    localStorage.setItem(PLAN_KEY, JSON.stringify(plan));
  }

  function loadPlan() {
    const raw = localStorage.getItem(PLAN_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    for (const p of arr) {
      if (!p.id) p.id = Date.now() + Math.random();
      if (!('donePages' in p)) p.donePages = 0;
    }
    return arr;
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

      const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = (done >= planned && planned>0);
      checkbox.addEventListener('change', ()=>{
        const currentDone = entries.filter(e=>e.date===p.date).reduce((s,e)=>s+Number(e.pages),0);
        if (checkbox.checked) {
          const need = p.pages - currentDone;
          if (need > 0) {
            addEntry(p.date, need, 'plan-auto', p.id);
          }
          p.donePages = Math.min(p.pages, currentDone + Math.max(0, p.pages-currentDone));
        } else {
          // remove auto-created entries for this plan id
          entries = entries.filter(e=>!(e.origin === 'plan-auto' && e.planId === p.id));
          save();
          p.donePages = entries.filter(e=>e.date===p.date).reduce((s,e)=>s+Number(e.pages),0);
          render();
        }
        savePlan(plan);
        renderPlan(plan);
        updatePlanSummary(plan);
      });

      const deleteBtn = document.createElement('button'); deleteBtn.className='delete-plan'; deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', ()=>{
        // remove plan and any auto-created entries linked to it
        const existing = loadPlan();
        const filtered = existing.filter(x=>x.id !== p.id);
        // remove auto-created entries for this plan id
        entries = entries.filter(e => !(e.origin === 'plan-auto' && e.planId === p.id));
        save();
        savePlan(filtered);
        renderPlan(filtered);
        updatePlanSummary(filtered);
        render();
      });

      right.appendChild(checkbox); right.appendChild(status); right.appendChild(deleteBtn);
      li.appendChild(left); li.appendChild(right);
      planList.appendChild(li);
    }
  }

  function updatePlanSummary(plan) {
    const totalPlanned = plan.reduce((s,p)=>s+Number(p.pages),0);
    const rawTarget = parseInt(targetInput.value, 10);
    const target = Number.isFinite(rawTarget) ? rawTarget : null;
    const totalWritten = entries.reduce((s,e)=>s+Number(e.pages),0);
    if (target) {
      const remainingToTarget = Math.max(0, target - totalWritten);
      const remainingToPlan = Math.max(0, remainingToTarget - totalPlanned);
      pagesToPlanEl.textContent = remainingToPlan;
    } else {
      // show total planned when no explicit target is set
      pagesToPlanEl.textContent = totalPlanned ? `${totalPlanned} planned` : '0';
    }
    pagesPerDayEl.textContent = plan.length ? (totalPlanned/plan.length).toFixed(1) : '0';
  }

  // Removed auto-schedule control; planner is manual only now.

  // Add manual plan item
  addPlanBtn.addEventListener('click', ()=>{
    const date = planDateInput.value || new Date().toISOString().slice(0,10);
    const pages = Math.max(0, parseInt(planPagesInput.value,10) || 0);
    if (!date) return alert('Pick a date');
    if (pages <= 0) return alert('Pages must be > 0');
    const plan = loadPlan();
    // if date exists, update pages
    const exists = plan.find(p=>p.date === date);
    if (exists) {
      exists.pages = pages;
      exists.manual = true;
    } else {
      // compute donePages from entries
      const done = entries.filter(e=>e.date===date).reduce((s,e)=>s+Number(e.pages),0);
      plan.push({ id: Date.now() + Math.random(), date, pages, donePages: done, manual: true });
      // sort by date
      plan.sort((a,b)=> new Date(a.date) - new Date(b.date));
    }
    savePlan(plan);
    renderPlan(plan);
    updatePlanSummary(plan);
    planPagesInput.value = '';
  });

  // load existing plan on startup — only show manually created plan items
  const existingPlan = loadPlan();
  const manualOnly = existingPlan.filter(p=>p.manual);
  renderPlan(manualOnly);
  updatePlanSummary(manualOnly);

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
  addEntry = function(date,pages,origin,planId){ oldAddEntry(date,pages,origin,planId); syncStoredPlanWithEntries(); };

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
