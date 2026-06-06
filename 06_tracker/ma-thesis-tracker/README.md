# Thesis Writing Tracker

Small single-file web app to track progress toward a page target (default 75 pages).

Usage

- Open [index.html](index.html) in your browser, or visit the live site: https://dariakiuru.github.io/writing-tracker/
- Or serve the folder and open it in the browser. From the `ma-thesis-tracker` folder run:

```powershell
# if you have Python installed
python -m http.server 8000

# then open http://localhost:8000 in your browser
```

Features

- Set target pages (default 75).
- Add daily entries (date + pages done).
- See total written, remaining pages, and a simple progress bar.
- Data is saved to `localStorage` in your browser.

Planner (manual)

- Pick a date, enter how many pages you plan to write that day, and click "Add plan entry".
- The plan list shows your planned pages per date. You can edit the planned pages inline or delete a plan item.
- When you check a plan item's checkbox it will create a matching entry and transfer the pages to the Entries pane. Unchecking removes auto-created entries for that plan item.
- The planner auto-syncs with any existing entries on load (so previously written pages are reflected in the plan status).
- Export the planned schedule as CSV or iCal (.ics) to add to your calendar.

Notes

- This is a simple client-side app; no server or authentication.
- If you want a GitHub-hosted site, you can publish the `ma-thesis-tracker` folder via GitHub Pages.
