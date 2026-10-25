"""Shared AirWatch design system, styled after shadcn/ui (zinc, dark).

Loaded ONCE by the entrypoint (Air.py). Pages must not inject their own CSS,
call set_page_config, or render the brand header.
"""
import streamlit as st


# ---------------------------------------------------------------
# Lucide icons (the icon set used by shadcn/ui), as inline SVG.
# icon("wind") returns a single-line <svg> that inherits text colour.
# ---------------------------------------------------------------
_PATHS = {
    "cloud-fog": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/><path d="M16 17H7"/><path d="M17 21H9"/>',
    "wind": '<path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>',
    "heart-pulse": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "hourglass": '<path d="M5 22h14"/><path d="M5 2h14"/><path d="M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22"/><path d="M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2"/>',
    "building-2": '<path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"/><path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"/><path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"/><path d="M10 6h4"/><path d="M10 10h4"/><path d="M10 14h4"/><path d="M10 18h4"/>',
    "microscope": '<path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>',
    "circle-check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "house": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/><path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
    "map-pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "sliders-horizontal": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "chart-column": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
}


def icon(name: str, size: int = 18) -> str:
    """Return a Lucide icon as an inline SVG string."""
    paths = _PATHS[name]
    return (
        f'<svg class="lucide" xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths}</svg>'
    )


CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* shadcn/ui "zinc" dark tokens */
:root {
    color-scheme: dark;
    --background: #09090b;
    --foreground: #fafafa;
    --card: #09090b;
    --muted: #27272a;
    --muted-foreground: #a1a1aa;
    --secondary: #27272a;
    --accent: #18181b;
    --border: #27272a;
    --input: #27272a;
    --ring: #d4d4d8;
    --primary: #fafafa;
    --primary-foreground: #18181b;
    --radius: 8px;
    --radius-lg: 12px;
}

html, body, [class*="css"] {
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--background) !important;
    color: var(--foreground);
}

header[data-testid="stHeader"] {
    background: rgba(9, 9, 11, 0.85);
    backdrop-filter: blur(8px);
}

#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4, h5 {
    color: var(--foreground) !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em;
}

p, li {
    color: var(--muted-foreground);
    line-height: 1.7;
    font-size: 0.925rem;
}

label, [data-testid="stWidgetLabel"] p {
    color: var(--foreground) !important;
    font-size: 0.875rem;
    font-weight: 500;
}

a { color: var(--foreground) !important; text-underline-offset: 4px; }

[data-testid="stCaptionContainer"] { color: #71717a !important; }

.text-muted { color: var(--muted-foreground); }

svg.lucide { flex-shrink: 0; vertical-align: -0.2em; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #09090b;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

[data-testid="stSidebarNavLink"] {
    border-radius: var(--radius);
    color: var(--muted-foreground);
    font-weight: 500;
}

[data-testid="stSidebarNavLink"]:hover {
    background: var(--accent);
    color: var(--foreground);
}

[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: var(--secondary);
    color: var(--foreground);
}

[data-testid="stSidebar"] hr { margin: 1rem 0; }
[data-testid="stSidebar"] h3 { font-size: 1.05rem; }

/* ---------- Brand ---------- */
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
}

.brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: var(--primary);
    color: var(--primary-foreground);
}

.brand-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--foreground);
    letter-spacing: -0.025em;
    line-height: 1.2;
}

.brand-subtitle {
    font-size: 0.78rem;
    color: var(--muted-foreground);
    margin-top: 2px;
}

/* ---------- Badge ---------- */
.eyebrow, .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 3px 11px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--foreground);
    background: transparent;
    line-height: 1.5;
}

.eyebrow { margin-bottom: 18px; }

.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

/* ---------- Hero ---------- */
.hero {
    padding: 40px;
    background:
        radial-gradient(ellipse at top right,
        rgba(255, 255, 255, 0.05), transparent 55%),
        var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -0.045em;
    color: var(--foreground);
    margin-bottom: 16px;
}

.hero-description {
    max-width: 660px;
    font-size: 1rem;
    color: var(--muted-foreground);
    line-height: 1.8;
}

/* ---------- Sections and cards ---------- */
.section-heading {
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: var(--foreground);
    margin-top: 2.6rem;
    margin-bottom: 4px;
}

.section-description {
    color: var(--muted-foreground);
    font-size: 0.875rem;
    margin-bottom: 20px;
}

.sub-heading {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--foreground);
    margin: 0.25rem 0 0.6rem;
}

.sub-heading svg { color: var(--muted-foreground); }

.info-card {
    height: 100%;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
    transition: border-color 0.15s ease;
}

.info-card:hover { border-color: #3f3f46; }

.card-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border-radius: var(--radius);
    background: var(--muted);
    color: var(--foreground);
    margin-bottom: 16px;
}

.card-title {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--foreground);
    margin-bottom: 8px;
}

.card-description {
    color: var(--muted-foreground);
    font-size: 0.875rem;
    line-height: 1.7;
}

.tip {
    display: flex;
    gap: 14px;
    align-items: flex-start;
}

.tip svg { color: var(--foreground); margin-top: 3px; }

[data-testid="stImage"] img {
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
}

/* AQI scale (prediction page) */
.scale-item {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
    background: var(--card);
}

.scale-item.active {
    border-color: var(--ring);
    background: var(--accent);
}

.scale-item .dot { margin-bottom: 8px; }
.scale-range { font-weight: 600; font-size: 0.9rem; color: var(--foreground); }
.scale-label { font-size: 0.78rem; color: var(--muted-foreground); }

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background: var(--background) !important;
    border: 1px solid var(--input) !important;
    border-radius: var(--radius) !important;
    color: var(--foreground) !important;
    min-height: 40px;
}

div[data-baseweb="select"] span,
input, textarea { color: var(--foreground) !important; }

input, textarea { background: transparent !important; }

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--ring) !important;
    box-shadow: 0 0 0 3px rgba(212, 212, 216, 0.15) !important;
}

span[data-baseweb="tag"] {
    background: var(--secondary) !important;
    color: var(--foreground) !important;
    border-radius: 6px !important;
}

div[data-testid="stNumberInput"] button {
    background: var(--accent) !important;
    color: var(--foreground) !important;
    border-color: var(--input) !important;
}

[data-baseweb="popover"] > div,
[data-baseweb="menu"],
ul[role="listbox"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

li[role="option"] {
    color: #e4e4e7 !important;
    background: transparent !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background: var(--accent) !important;
    color: var(--foreground) !important;
}

/* ---------- Buttons (shadcn "default" variant) ---------- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    width: 100%;
    min-height: 40px;
    background: var(--primary) !important;
    color: var(--primary-foreground) !important;
    border: 1px solid var(--primary) !important;
    border-radius: var(--radius);
    font-weight: 500;
    font-size: 0.875rem;
    box-shadow: none !important;
    transition: background 0.15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: rgba(250, 250, 250, 0.9) !important;
    border-color: transparent !important;
    color: var(--primary-foreground) !important;
}

.stButton button p,
.stDownloadButton button p,
.stFormSubmitButton button p,
.stFormSubmitButton button span {
    color: var(--primary-foreground) !important;
    font-weight: 500;
}

button:focus-visible, input:focus-visible {
    outline: 2px solid var(--ring) !important;
    outline-offset: 2px;
}

/* ---------- Metrics (Card) ---------- */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    min-height: 110px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}

[data-testid="stMetricLabel"] {
    color: var(--muted-foreground) !important;
    font-size: 0.8rem;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    color: var(--foreground) !important;
    font-weight: 700;
    font-size: 1.8rem;
    letter-spacing: -0.03em;
}

/* ---------- Form, expander, alert ---------- */
[data-testid="stForm"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border);
    border-radius: var(--radius-lg);
}

[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
}

[data-testid="stAlert"],
[data-baseweb="notification"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--foreground) !important;
}

hr {
    border-color: var(--border) !important;
    margin-top: 1.75rem;
    margin-bottom: 1.75rem;
}

/* ---------- Footer ---------- */
.footer {
    border-top: 1px solid var(--border);
    padding-top: 24px;
    margin-top: 48px;
    color: #71717a;
    font-size: 0.8rem;
    line-height: 1.8;
}

.footer strong {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #d4d4d8;
    font-weight: 600;
}

/* ---------- Anti-flicker ---------- */
[data-stale="true"],
[data-testid="stElementContainer"][data-stale="true"],
.element-container[data-stale="true"] {
    opacity: 1 !important;
    transition: none !important;
}

[data-testid="stElementContainer"],
.element-container { transition: none !important; }

[data-testid="stStatusWidget"] { visibility: hidden; }

[data-testid="stElementContainer"]:has(style),
.element-container:has(style) { display: none; }

/* ---------- Smooth entrance (opacity only) ---------- */
@keyframes aw-fade {
    from { opacity: 0; }
    to   { opacity: 1; }
}

.hero, .info-card, [data-testid="stMetric"],
[data-testid="stForm"], [data-testid="stImage"] {
    animation: aw-fade 0.35s ease-out both;
}

@media (prefers-reduced-motion: reduce) {
    .hero, .info-card, [data-testid="stMetric"],
    [data-testid="stForm"], [data-testid="stImage"] { animation: none; }
}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    .block-container { padding: 1.2rem 1rem 3rem; }
    .hero { padding: 24px; }
    .hero-title { font-size: 2.1rem; }
}
</style>"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def render_brand():
    st.markdown(
        f"""
<div class="brand">
    <div class="brand-icon">{icon("cloud-fog", 22)}</div>
    <div>
        <div class="brand-name">AirWatch India</div>
        <div class="brand-subtitle">Air quality intelligence platform</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def sub_heading(icon_name: str, text: str) -> str:
    """HTML for a small icon + title heading (use with unsafe_allow_html)."""
    return f'<div class="sub-heading">{icon(icon_name, 18)}<span>{text}</span></div>'