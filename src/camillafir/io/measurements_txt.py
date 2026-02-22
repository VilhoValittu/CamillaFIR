import os
import numpy as np


def parse_measurements_from_bytes(file_content):
    """
    Jasentaa REW-tyylisen mittausdatan tavu- tai merkkijonosyotteesta.

    Hyvaksyy rivit, joissa on vahintaan taajuus ja amplitudi, sekavia
    desimaalierottimia (pilkku/piste) seka valinnaisen vaihesarakkeen.
    Palauttaa `(freqs, mags, phases)` tai `(None, None, None)`, jos dataa
    ei voida lukea.
    """
    try:
        if isinstance(file_content, (bytes, bytearray)):
            content_str = file_content.decode("utf-8", errors="ignore")
        else:
            content_str = str(file_content)

        lines = content_str.splitlines()
        freqs, mags, phases = [], [], []

        for line in lines:
            line = line.strip()
            if not line or line.startswith(("*", "#", ";")):
                continue
            if not line[0].isdigit() and line[0] != "-":
                continue

            if "," in line and "." in line:
                line = line.replace(",", " ")
            else:
                line = line.replace(",", ".")

            parts = line.split()
            if len(parts) >= 2:
                try:
                    f_val = float(parts[0])
                    m_val = float(parts[1])
                    p_val = float(parts[2]) if len(parts) > 2 else 0.0
                    freqs.append(f_val)
                    mags.append(m_val)
                    phases.append(p_val)
                except ValueError:
                    continue

        if len(freqs) == 0:
            return None, None, None

        return np.array(freqs), np.array(mags), np.array(phases)

    except Exception:
        return None, None, None


def parse_measurements_from_path(path, *, logger=None):
    """
    Lukee mittausdatan paikallisesta tekstitiedostosta ja jasentaa sen.

    Polku siistitään turvallisesti, olemassaolo tarkistetaan ennen lukua ja
    mahdolliset virheet kirjataan loggeriin, jos logger on annettu.
    """
    try:
        if not path:
            return None, None, None

        p = str(path).strip().strip('"').strip("'")
        if not os.path.exists(p):
            if logger:
                logger.error(f"File not found: {p}")
            return None, None, None

        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return parse_measurements_from_bytes(content)

    except Exception as e:
        if logger:
            logger.error(f"Kriittinen virhe polun luvussa ({path}): {e}")
        return None, None, None
