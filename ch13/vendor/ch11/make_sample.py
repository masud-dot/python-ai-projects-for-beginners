"""Write docs/harbor_bikes.pdf, a made-up four-page handbook."""
import textwrap
from pathlib import Path

PAGES = [
    "Harbor Street Bike Share: Rider Handbook. This handbook "
    "is invented for the exercises in this book. Membership. "
    "A day pass costs 6 dollars and covers unlimited rides of "
    "up to 30 minutes each for 24 hours. A monthly membership "
    "costs 25 dollars and covers rides of up to 45 minutes. "
    "An annual membership costs 220 dollars and also covers "
    "rides of up to 45 minutes. Students with a valid school "
    "card get 20 percent off the monthly and annual prices. "
    "Overtime. Each extra 15 minutes past the ride limit costs "
    "3 dollars. Payment. Memberships renew on the same date "
    "each period. A card payment that fails and is not fixed "
    "within 7 days gets a 5 dollar late fee.",
    "Riding rules. Riders must be at least 16 years old. "
    "Helmets are recommended but not required. A bike carries "
    "one rider and no passengers. Ride on streets and in "
    "marked bike lanes. In the downtown zone, riding on the "
    "sidewalk is not allowed. Lights switch on by themselves "
    "at dusk. The front basket holds up to 10 kilograms. "
    "Service zone. Bikes must stay inside the service zone, "
    "which ends at the Mill River bridge on the east side and "
    "at Pine Avenue on the west side. A rider who leaves the "
    "zone may be charged the overtime rate until the bike is "
    "back inside it.",
    "Returning a bike. To end a ride, push the bike into any "
    "dock at a station and wait for the dock light to turn "
    "green. If the light is red, the dock is faulty, so try "
    "another dock. Damage. If a bike is damaged, press the "
    "report button on the handlebar screen or call the help "
    "line. Riders are not charged for normal wear. Late "
    "returns. A bike kept overnight but returned within 24 "
    "hours is charged a 40 dollar late return fee. A bike not "
    "returned within 24 hours is treated as lost, and the "
    "rider is charged a 400 dollar replacement fee.",
    "Help and refunds. The help line is staffed from 7 am to "
    "9 pm every day. Email help@harborbikes.example for "
    "questions that are not urgent. Refunds. Day passes "
    "cannot be refunded. A monthly membership can be "
    "cancelled at any time, and full unused weeks are "
    "refunded. An annual membership can be refunded within "
    "14 days of purchase. Lost property. Items left in a "
    "basket are held for 30 days at the Central Station "
    "office. Stations are cleaned every Monday morning.",
]


def escape(line):
    """Escape the characters that are special in PDF strings."""
    return (line.replace("\\", "\\\\").replace("(", "\\(")
            .replace(")", "\\)"))


def page_stream(text):
    """Content stream: 11-point Helvetica, 80 characters."""
    lines = textwrap.wrap(text, width=80)
    ops = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
    ops += [f"({escape(line)}) '" for line in lines]
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def write_pdf(pages, path):
    """Write a plain text PDF with one string per page."""
    count = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>"
        .encode(),
    ]
    font = 3 + 2 * count
    for i, text in enumerate(pages):
        stream = page_stream(text)
        objects.append(
            (f"<< /Type /Page /Parent 2 0 R /MediaBox "
             f"[0 0 612 792] /Contents {4 + 2 * i} 0 R "
             f"/Resources << /Font << /F1 {font} 0 R >> >> >>")
            .encode()
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream + b"\nendstream"
        )
    objects.append(
        b"<< /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >>"
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body
        out += b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R "
            f">>\nstartxref\n{start}\n%%EOF\n").encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(bytes(out))


if __name__ == "__main__":
    write_pdf(PAGES, "docs/harbor_bikes.pdf")
    print("wrote docs/harbor_bikes.pdf")
