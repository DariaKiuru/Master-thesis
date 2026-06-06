# MA Thesis Pages Tracker

Small single-file web app to track progress toward a page target (default 75 pages).

Usage

- Open [ma-thesis-tracker/index.html](ma-thesis-tracker/index.html) in your browser.
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

Notes

- This is a simple client-side app; no server or authentication.
- If you want a GitHub-hosted site, you can publish the `ma-thesis-tracker` folder via GitHub Pages.
