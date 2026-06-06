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
  }

  // init
  load(); setToday(); render();
})();
