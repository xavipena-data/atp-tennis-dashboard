"""Build dashboard/index.html by embedding the dashboard data into the template."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "dashboard" / "template.html"
DATA = ROOT / "data" / "processed" / "dashboard_data.json"
OUTPUT = ROOT / "dashboard" / "index.html"

# The template holds this exact text where the data goes
MARKER = "/*__DASHBOARD_DATA__*/null"


def main():
    template = TEMPLATE.read_text(encoding="utf-8")
    data = DATA.read_text(encoding="utf-8")

    if template.count(MARKER) != 1:
        raise SystemExit(f"Expected the marker exactly once in {TEMPLATE.name}")

    OUTPUT.write_text(template.replace(MARKER, data), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {OUTPUT.stat().st_size / 1024:,.0f} KB")


if __name__ == "__main__":
    main()