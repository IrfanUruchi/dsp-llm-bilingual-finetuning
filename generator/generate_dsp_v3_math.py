#!/usr/bin/env python
import json
import math
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "generated" / "dsp_albanian_v3_math.jsonl"
random.seed(42)


def make_fft_resolution_example():
    Fs = random.choice([8000, 10000, 16000, 22050, 44100])
    N = random.choice([256, 512, 1024, 2048, 4096])
    df = Fs / N
    df_rounded = round(df, 3)

    user = (
        f"Për Fs = {Fs} Hz dhe N = {N}, gjej rezolucionin në frekuencë Δf të FFT-së "
        f"dhe shpjego shkurt çfarë do të thotë ky rezolucion."
    )

    assistant = (
        "Rezolucioni në frekuencë i FFT-së tregon largësinë mes dy bin-eve fqinj në spektër.\n\n"
        f"Formula është:\n"
        f"Δf = Fs / N\n"
        f"Δf = {Fs} / {N} ≈ {df_rounded} Hz\n\n"
        f"Kjo do të thotë se çdo bin i FFT-së përfaqëson rreth {df_rounded} Hz diferencë "
        "në frekuencë. Pra spektrogrami mund të dallojë komponentë që ndryshojnë të paktën kaq."
    )

    return user, assistant


def make_fft_bin_example():
    Fs = random.choice([8000, 10000, 16000, 22050, 44100])
    N = random.choice([256, 512, 1024, 2048, 4096])
    k = random.randint(1, N // 4) 
    df = Fs / N
    df_r = round(df, 4)
    fk = k * df
    fk_r = round(fk, 2)

    user = (
        f"Për një FFT me Fs = {Fs} Hz dhe N = {N}, gjej rezolucionin Δf dhe "
        f"frekuencën e bin-it k = {k}."
    )

    assistant = (
        "Së pari llogarisim rezolucionin në frekuencë të FFT-së:\n\n"
        f"Δf = Fs / N = {Fs} / {N} ≈ {df_r} Hz\n\n"
        "Frekuenca e bin-it k llogaritet si:\n"
        "fₖ = k · Δf\n"
        f"fₖ = {k} · {df_r} ≈ {fk_r} Hz\n\n"
        f"Pra ky bin përfaqëson një komponent të sinjalit rreth frekuencës {fk_r} Hz."
    )

    return user, assistant


def make_nyquist_example():
    fmax = random.choice([1000, 1500, 3000, 4000, 5000])
    Fs_ok = 2 * fmax
    Fs_bad = int(Fs_ok * random.uniform(0.4, 0.9))

    user = (
        f"Një sinjal ka frekuencë maksimale rreth {fmax} Hz. "
        "Cila është frekuenca minimale e mostrimit sipas teoremës së Nyquist-it "
        "dhe çfarë ndodh nëse mostrojmë me një frekuencë më të ulët, p.sh. "
        f"Fs = {Fs_bad} Hz?"
    )

    assistant = (
        "Teorema e Nyquist-it thotë se për të shmangur aliasing-un duhet të mostrojmë "
        "me frekuencë të paktën dy herë më të madhe se frekuenca maksimale e sinjalit.\n\n"
        "Pra:\n"
        "Fs_min = 2 · f_max\n"
        f"Fs_min = 2 · {fmax} Hz = {Fs_ok} Hz\n\n"
        f"Nëse përdorim Fs = {Fs_bad} Hz < {Fs_ok} Hz, sinjali nuk plotëson kriterin e Nyquist-it "
        "dhe do të shfaqet aliasing: komponentët me frekuenca mbi Fs/2 do të projektohen gabimisht "
        "në frekuenca më të ulëta në spektër."
    )

    return user, assistant


def make_snr_example():
    Ps = random.choice([0.5, 1.0, 2.0, 5.0, 10.0]) 
    ratio = random.choice([2, 5, 10, 20])     
    Pn = Ps / ratio
    snr_lin = Ps / Pn
    snr_db = 10 * math.log10(snr_lin)
    Ps_r = round(Ps, 3)
    Pn_r = round(Pn, 3)
    snr_db_r = round(snr_db, 2)

    user = (
        f"Një sinjal ka fuqi mesatare Ps = {Ps_r} (njësi arbitrare), "
        f"ndërsa zhurma ka fuqi Pn = {Pn_r}. Llogarit SNR në dB dhe shpjego shkurt "
        "çfarë tregon ky numër."
    )

    assistant = (
        "Raporti sinjal-zhurmë (SNR) në dB jepet nga formula:\n\n"
        "SNR_dB = 10 · log10(Ps / Pn)\n\n"
        f"Ps = {Ps_r}, Pn = {Pn_r} → Ps / Pn = {snr_lin:.3f}\n"
        f"SNR_dB = 10 · log10({snr_lin:.3f}) ≈ {snr_db_r} dB\n\n"
        f"Kjo do të thotë se fuqia e sinjalit është rreth {snr_db_r} dB mbi fuqinë e zhurmës. "
        "Sa më i madh të jetë SNR në dB, aq më i qartë konsiderohet sinjali."
    )

    return user, assistant


def make_language_polish_example():
    bad_sentences = [
        "Rezolucioni i mostres është kur FFT është 1024 dhe bëhet frekuenca e shkojë poshtë e lartë.",
        "Aliasingu ndodh kur sinjali është marrë keq dhe dhe frekuencat kthehen mbrapsht në spektër.",
        "Nyquisti thotë që duhet të marrësh shumë mostra që sinjali të mos prishet në grafik.",
    ]
    good_sentences = [
        "Rezolucioni në frekuencë i FFT-së përcaktohet nga raporti Fs / N dhe tregon largësinë "
        "mes dy frekuencave fqinj që mund të dallojë analiza.",
        "Aliasingu ndodh kur mostrojmë me një frekuencë më të ulët se dyfishi i frekuencës maksimale "
        "të sinjalit; atëherë komponentët e lartë në frekuencë ‘palosen’ në zona më të ulëta.",
        "Teorema e Nyquist-it thotë se frekuenca e mostrimit duhet të jetë të paktën dy herë më e "
        "madhe se frekuenca maksimale e sinjalit, në mënyrë që sinjali të rikonstruktohet pa aliasing.",
    ]

    idx = random.randrange(len(bad_sentences))
    bad = bad_sentences[idx]
    good = good_sentences[idx]

    user = (
        "Më poshtë ke një fjali teknike në shqip për DSP, por është e paqartë ose jo shumë e saktë.\n"
        "Riformuloje në shqip të pastër dhe teknikisht të saktë:\n\n"
        f"\"{bad}\""
    )

    assistant = good
    return user, assistant


def build_chat(user, assistant):
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Je një asistent ekspert në Përpunimin Dixhital të Sinjaleve (DSP). "
                    "Përgjigju vetëm në gjuhën shqipe, qartë dhe teknikisht saktë."
                ),
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_fft_res = 120
    n_fft_bin = 120
    n_nyquist = 120
    n_snr = 80
    n_lang = 60

    with OUT_PATH.open("w", encoding="utf-8") as f:
       
        for _ in range(n_fft_res):
            u, a = make_fft_resolution_example()
            f.write(json.dumps(build_chat(u, a), ensure_ascii=False) + "\n")

        for _ in range(n_fft_bin):
            u, a = make_fft_bin_example()
            f.write(json.dumps(build_chat(u, a), ensure_ascii=False) + "\n")

  
        for _ in range(n_nyquist):
            u, a = make_nyquist_example()
            f.write(json.dumps(build_chat(u, a), ensure_ascii=False) + "\n")

        for _ in range(n_snr):
            u, a = make_snr_example()
            f.write(json.dumps(build_chat(u, a), ensure_ascii=False) + "\n")

        for _ in range(n_lang):
            u, a = make_language_polish_example()
            f.write(json.dumps(build_chat(u, a), ensure_ascii=False) + "\n")

    print(f"Wrote dataset to: {OUT_PATH}")


if __name__ == "__main__":
    main()
