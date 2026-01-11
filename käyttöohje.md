# CamillaFIR (v2.7.0 Stable) – Käyttöopas

CamillaFIR on tekoälyavusteinen DSP-moottori korkean resoluution FIR-suodattimien luomiseen. Ohjelma analysoi huoneakustiikan ja korjaa taajuusvasteen lisäksi ajoitusvirheitä ja resonansseja (huonemoodeja).

---

## Mitä uutta versiossa 2.7.0?

1. **Älykäs Ikkunointi (Tukey-ikkuna)**: `Asymmetric`- ja `Minimum Phase` -suodattimet käyttävät nyt Tukey-ikkunaa. Tämä säilyttää impulssivasteen alun energian ja transientit huomattavasti perinteistä Hann-ikkunaa paremmin, mikä kuuluu tarkempana ja dynaamisempana äänenä.
2. **Aito Akustinen Analyysi**: Ohjelma poistaa kaiuttimen etäisyysviiveen (TOF) ennen analyysia. Tämän ansiosta Summary-raportin "Acoustic Events" -kohdassa näkyvät etäisyydet (esim. 1.2m tai 0.8m) ovat todellisia huoneheijastuksia pinnoista.
3. **Päivitetty Dashboard**: 
   - **Level Match Range**: Harmaa alue magnitudikuvaajassa osoittaa taajuusalueen, jolla kaiuttimen taso on sovitettu tavoitekäyrään.
   - **Measured (Clean)**: Mitattu vaste on nyt psykoakustisesti tasoitettu, mikä tekee kuvaajasta selkeämmän lukea.

---

## Tekninen huomio: Suodattimen toiminta

Saatat havaita suodattimessa pientä väreilyä myös asetettujen rajojen yläpuolella. Tämä johtuu seuraavista syistä:
1. **Globaali Offset**: Koko suodattimen tasoa siirretään (esim. -70 dB), jotta suuret bassokorotukset eivät aiheuta digitaalista säröä. Dashboardilla viivat on kohdistettu vastaamaan toisiaan visuaalisesti.
2. **Teoreettinen vaihe**: Jakosuotimien (Crossovers) vaihekorjaus lasketaan aina koko kaistalle (20 kHz asti), jotta ajoitus säilyy eheänä.
3. **Soft Clip**: `max_boost` -arvoa noudatetaan pehmeällä vaimennuksella (Soft Clipping), joka estää äkilliset leikkaukset äänessä, vaikka tavoite vaatisi enemmän vahvistusta.

---

## Summary: Täydellinen Dashboard (v2.7.0)

1. **Magnitude**: Oranssi ennusteviiva (`Predicted`) seuraa vihreää tavoitetta korjausalueella. Harmaa kaistale osoittaa, missä taso on täsmätty.
2. **Measured (Clean)**: Sininen viiva on siisti ja se kohtaa tavoitekäyrän täsmälleen harmaan alueen sisällä.
3. **Group Delay**: Bassopään piikit on tasoitettu, ja Summary-raportti näyttää järkeviä etäisyyksiä (senttimetrejä tai metrejä) huoneen heijastuksille.
4. **Filter dB**: Purppura viiva näyttää suodattimen tekemän työn. Se pysyy hallittuna `max_boost` -asetuksen puitteissa.

## Akustinen analyysi ja toimenpiteet (Acoustic Intelligence)

Versiosta 2.7.0 alkaen ohjelman ilmoittamat etäisyydet ovat erittäin tarkkoja, koska kaiuttimen oma etäisyysviive (TOF) on poistettu laskennasta. Summary-raportin "Acoustic Events" -lista auttaa sinua ymmärtämään huoneesi ongelmia:

1. **Resonance (Resonanssi)**:
   - Näitä esiintyy tyypillisesti alle 200 Hz taajuuksilla.
   - Ne ovat huoneen seisovia aaltoja (huonemoodeja).
   - **Toimenpide**: Käytä TDC (Temporal Decay Control) -toimintoa vaimentamaan näiden soimisaikaa. Mitä suurempi "Virhe (ms)", sitä tärkeämpää korjaus on.

2. **Reflection (Heijastus)**:
   - Näitä esiintyy yleensä yli 200 Hz taajuuksilla.
   - Raportin ilmoittama "Etäisyys" kertoo, kuinka monta metriä heijastunut ääni on kulkenut kaiuttimesta pintaan ja takaisin mikrofoniin.
   - **Esimerkki**: Jos näet heijastuksen 1.2 metrin kohdalla, etsi heijastavaa pintaa (kuten työpöytä, lattia tai sivuseinä) noin 60 cm etäisyydeltä kaiuttimesta.
   - **Toimenpide**: Voit yrittää vaimentaa näitä akustointilevyillä tai muuttamalla kaiuttimen/kuuntelupaikan sijoitusta.

3. **Luottamusindeksi (Confidence %)**:
   - Kertoo, kuinka hyvin DSP-moottori pystyy erottamaan suoran äänen huoneen hälystä.
   - Yli 80 % lukema on erinomainen. Jos lukema on matala tietyllä taajuudella, huoneessa on kyseisellä kohdalla erittäin voimakas ja monimutkainen heijastus, jota on vaikea korjata täydellisesti ilman fyysistä akustointia.


---

## 1. Perusasetukset (Input & Technical)
Määritellään lähdemateriaali ja suodattimen tekniset raamit.

* **Local Path L / R**: Valitse REW-ohjelmalla viedyt `.txt` tai `.frd` tiedostot (Export measurement as text). Varmista, että tiedostossa on mukana sekä magnitudi (dB) että vaihe (Phase).
* **Sample Rate (Hz)**: CamillaDSP-järjestelmäsi näytteenottotaajuus (yleensä 44100 tai 48000).
* **Number of Taps**: Suodattimen pituus.
    * **65536**: Standardi, erinomainen tarkkuus.
    * **131072**: Äärimmäinen tarkkuus bassoalueelle, vaatii enemmän CPU-tehoa.
* **Output Format**: `WAV` (standardi CamillaDSP:lle) tai `TXT`.



---

## 2. Suodatintyyppi (Filter Design)
Määrittää, miten ohjelma käsittelee äänen ajoitusta (vaihetta).

* **Filter Type**:
    * **Linear Phase**: Täydellinen vaiheenkorjaus. Napakka isku, mutta voi aiheuttaa esivärähtelyä (pre-ringing).
    * **Minimum Phase**: Ei korjaa vaihetta. Luonnollisin diskantti, ei esikaikua.
    * **Asymmetric Linear**: Optimivaihtoehto. Hallittu esisointi, säilyttää terävät iskuäänet.
    * **Mixed Phase**: Basso korjataan lineaarisesti (napakkuus) ja diskantti minimivaiheisesti (ilmavuus).
* **Mixed Phase Split (Hz)**: Taajuus, jolla vaihdetaan vaiheenkäsittelyä (suositus: 300–500 Hz).
* **Global Gain (dB)**: Koko suodattimen voimakkuus. Pidetään yleensä 0:ssa automaattisen normalisoinnin vuoksi.



---

## 3. Tavoitekäyrä (Target & Magnitude)
Määrittää järjestelmän lopullisen sointitasapainon.

* **House Curve Mode**:
    * **Flat**: Täysin suora vaste.
    * **Harman (+6dB / +8dB)**: Suosituin asetus. Nostaa bassoa ja laskee diskanttia pehmeästi.
* **Correction Range (Min/Max)**: Alue, jolla korjaus tapahtuu (esim. 20 Hz – 500 Hz tai koko kaista).
* **Max Boost (dB)**: Kuinka paljon ohjelma saa nostaa vaimentumia. Suositus: 6–8 dB.



---

## 4. Voimakkuuden tasaus (Leveling)
Varmistaa, ettei suodatin muuta äänenvoimakkuutta merkittävästi.

* **Level Mode**: `Automatic` (suositus) tai `Manual`.
* **Match Range (Min/Max)**: Alue, jolta äänenpaineen keskiarvo lasketaan (suositus: 500–2000 Hz).
* **Algorithm**: `Median` sivuuttaa yksittäiset piikit paremmin kuin `Average`.

---

## 5. Laskenta & Tasoitus (DSP & Smoothing)
Määrittää korjauksen "tiukkuuden".

* **Regulation Strength**:
    * **1–10**: Erittäin tiukka seuranta (veitsenterävä basso). *Huom: v2.6.5--> versiossa asetus 30 vastaa aiempaa 1-arvoa.*
    * **50+**: Pehmeämpi korjaus, säästää kaiuttimen luonnetta.
* **Smoothing Type**: `Psychoacoustic` (suositus) vastaa ihmiskorvan aistimusta.
* **FDW Cycles**: Aikaikkunan pituus. Suositus: 15.

---

## 6. Älykäs korjaus (Smart Correction)
Ohjelman edistyneet analyysiominaisuudet.

* **Adaptive FDW (A-FDW)**: Säätää ikkunointia dynaamisesti mittauksen luotettavuuden mukaan.
* **Temporal Decay Control (TDC)**:
    * **Toiminta**: Vähentää huoneresonanssien soimisaikaa (kumu).
    * **Strength**: 50–80 % poistaa basson "hännät" tehokkaasti.



---

## 7. Jakosuotimet & Suojat (Crossovers & Protection)

* **HPF (High Pass Filter)**: Suojaa kaiuttimia alimmilta taajuuksilta (esim. 20 Hz / 24dB).
* **Crossovers (1-5)**: Mahdollistaa monitiesuodinten rakentamisen automaattisella vaiheenkorjauksella.
* **Excursion Protection**: Estää sähköisen noston (boost) valitun taajuuden alapuolella elementin suojaamiseksi.

---

## 8. Lisävalinnat (Advanced)

* **Stereo Link**: Pakottaa saman voimakkuuskorjauksen molemmille kanaville (keskikuvan säilyminen).
* **Alignment**: Varmistaa impulssin huipun täydellisen ajoituksen.
* **Multi-rate**: Generoi suodattimet automaattisesti useille eri näytteenottotaajuuksille.

---

## 9. Dashboardin tulkitseminen (Visual Analytics)

Dashboard on interaktiivinen HTML-näkymä, joka näyttää suodattimen vaikutuksen neljässä eri paneelissa. Se on jaettu akustiseen analyysiin ja sähköiseen suodatinvasteeseen.



### A. Magnitude & Confidence (Ylin paneeli)
Tämä on tärkein näkymä vasteen tasaisuuden tarkistamiseen.
* **Sininen pilkkuviiva**: Alkuperäinen mittauksesi.
* **Vihreä viiva**: Valittu tavoitekäyrä (House Curve).
* **Oranssi viiva**: **Predicted** – miltä vaste näyttää suodattimen jälkeen. Tämän tulisi seurata vihreää viivaa mahdollisimman tarkasti.
* **Magenta viiva (Confidence)**: Kertoo mittauksen luotettavuuden (0-100%).
    * *Yli 85%*: Erinomainen. Suodatin voi korjata piikit tarkasti.
    * *Alle 50%*: Mittauksessa on liikaa heijastuksia tai häiriöitä. Ohjelma vähentää korjausta automaattisesti näillä alueilla suojellakseen äänenlaatua.

### B. Phase (Vaihe)
Kertoo äänen ajoituksesta eri taajuuksilla.
* **Tavoite**: Mahdollisimman suora viiva valittuun vaiherajaan asti (Phase Limit).
* **Merkitys**: Suora vaihe tarkoittaa, että kaiuttimen eri elementit ja huoneen ensiheijastukset on tahdistettu. Tämä parantaa stereokuvan tarkkuutta ja "kolmiulotteisuutta".

### C. Group Delay (Ryhmäviive)
Paljastaa, kuinka paljon huonemoodit (resonanssit) viivästyttävät ääntä.
* **Piikit kuvaajassa**: Kertovat, että tietyt taajuudet "jäävät soimaan" huoneeseen. Esimerkiksi 32 Hz piikki tarkoittaa, että basso kumisee kyseisellä taajuudella.
* **Tavoite**: Oranssin viivan tulisi olla mahdollisimman matalalla ja tasainen verrattuna siniseen pilkkuviivaan.



### D. RT60 (Jälkikaiunta-aika)
Kertoo, kuinka nopeasti ääni vaimenee huoneessasi.
* **Hyvä arvo**: Kotihuoneessa yleensä 0.2s – 0.5s.
* **CamillaFIR-hyöty**: Ohjelma käyttää tätä tietoa **Smart TDC** -logiikassa. Jos jokin taajuus soi huomattavasti pitempään kuin RT60-keskiarvo, suodatin puree siihen aggressiivisemmin.

### E. Filter Magnitude (Alin paneeli)
Tämä näyttää itse FIR-suodattimen "muodon".
* **Huomio**: Jos näet tässä suuria, kapeita piikkejä ylöspäin, olet asettanut **Max Boost** -arvon ehkä liian korkeaksi.
* **Normalisointi**: Kuvaaja on yleensä nollarajan alapuolella. Tämä on normaalia ja varmistaa, ettei digitaalinen särö (clipping) riko ääntä.



---

## 10. Pikavinkit ongelmatilanteisiin

| Havainto Dashboardissa | Syy | Ratkaisu |
| :--- | :--- | :--- |
| Oranssi viiva on kaukana vihreästä | Max Boost on liian pieni | Nosta Max Boost (esim. 6 -> 8 dB) |
| Bassoalueella on suuria mutkia | Regulation Strength on liian suuri | Laske regulaatio arvoon 10-30 |
| Phase-kuvaaja on sekava yläpäässä | Phase Limit on liian korkea | Laske Phase Limit arvoon 400-600 Hz |
| Confidence-viiva on hyvin matala | Taustahälyä tai huono mikrofoni | Tee uusi mittaus hiljaisemmassa huoneessa |


# 11. Vianetsintä-opas: Akustiikan tulkinta Dashboardilla

Tämä osio auttaa sinua tunnistamaan yleisimmät akustiikkaongelmat Dashboardin kuvaajista ja kertoo, miten voit vaikuttaa niihin CamillaFIR-asetuksilla.

---

## 1. Ongelma: Hallitsematon bassoresonanssi (Room Mode)

**Miltä se näyttää Dashboardilla?**
* **Magnitude**: Korkea ja kapea piikki bassoalueella (esim. 30–100 Hz).
* **Group Delay**: Terävä, korkea torni samalla taajuudella.
* **RT60**: Arvo nousee kyseisellä taajuudella huomattavasti muuta huonetta korkeammalle.



**Ratkaisu CamillaFIR:ssä:**
1.  **Regulation Strength**: Laske arvoon 10–30 (v2.6.3 skaalalla). Tämä pakottaa suodattimen leikkaamaan piikin tarkasti.
2.  **Temporal Decay Control (TDC)**: Nosta voimakkuutta (Strength 70–90%). TDC on suunniteltu nimenomaan pysäyttämään resonanssin soiminen.

---

## 2. Ongelma: Akustinen vaimenuma (Acoustic Null)

**Miltä se näyttää Dashboardilla?**
* **Magnitude**: Syvä, kapea "monttu", jota oranssi ennusteviiva ei pysty täyttämään, vaikka se yrittää.
* **Confidence**: Viiva putoaa usein montun kohdalla.



**Analyysi:**
Kyseessä on yleensä heijastus, jossa ääni kumoaa itsensä. Tätä ei voi korjata lisäämällä tehoa (DSP-nosto), koska lisäteho aiheuttaa vain voimakkaamman heijastuksen ja kumoamisen.

**Ratkaisu CamillaFIR:ssä:**
1.  **Älä yritä pakottaa**: Pidä **Max Boost** maltillisena (max 6-8 dB).
2.  **Sijoittelu**: Jos monttu on kriittisellä alueella, kokeile siirtää kaiuttimia tai kuuntelupaikkaa 10–20 cm. DSP ei voi voittaa fysiikkaa tässä tapauksessa.

---

## 3. Ongelma: Esivärähtely (Pre-ringing)

**Miltä se näyttää Dashboardilla?**
* **Korva**: Kuulet "sähisevän" tai "metallisen" kaiun ennen kovaa iskuääntä (esim. virvelirumpu).
* **Impulse Response**: Näet värähtelyä ennen pääimpulssin huippua.



**Ratkaisu CamillaFIR:ssä:**
1.  **Filter Type**: Vaihda `Asymmetric Linear` tai `Mixed Phase` -suodattimeen.
2.  **Phase Limit**: Laske vaiheenkorjauksen rajaa (esim. 400 Hz). Mitä korkeammalle korjaat vaihetta, sitä suurempi riski esisoinnille on.

---

## 4. Ongelma: Kampasuodin-ilmiö (Comb Filtering)

**Miltä se näyttää Dashboardilla?**
* **Magnitude**: Vaste siksakkaa tiheästi ylös ja alas keski- ja ylätaajuuksilla.
* **Phase**: Vaihekuvaaja pyörii villisti ympäri.

**Analyysi:**
Tämä johtuu yleensä voimakkaasta heijastuksesta läheisestä pinnasta (esim. työpöytä tai sivuseinä).

**Ratkaisu CamillaFIR:ssä:**
1.  **Smoothing**: Varmista, että käytät `Psychoacoustic` -tasoitusta, jotta ohjelma ei yritä korjata näitä mikroskooppisia virheitä liian aggressiivisesti.
2.  **FDW Cycles**: Pienennä arvoa (esim. 15 -> 10). Lyhyempi aikaikkuna sivuuttaa myöhemmin saapuvat heijastukset.

---

## 5. Ongelma: Epätarkka keskikuva (Phase Mismatch)

**Miltä se näyttää Dashboardilla?**
* **Vasen vs. Oikea**: Kun vertaat L- ja R-kanavien Phase-paneeleita, ne näyttävät täysin erilaisilta 200–1000 Hz alueella.

**Ratkaisu CamillaFIR:ssä:**
1.  **Stereo Link**: Varmista, että magnitudikorjaus on linkitetty, jos huone on symmetrinen.
2.  **Phase Limit**: Varmista, että molemmilla kanavilla on sama vaiheraja, jotta ajoitus korjataan identtisesti.

---

## Yhteenveto: Ideaali Dashboard

1.  **Magnitude**: Oranssi viiva seuraa vihreää +/- 2 dB tarkkuudella.
2.  **Group Delay**: Bassoalueen piikit on saatu tasoitettua alkuperäisestä.
3.  **Confidence**: Pysyy yli 80% lähes koko taajuusalueella.
4.  **RT60**: Kuvaaja on tasainen ilman yksittäisiä "piikkejä" tietyillä taajuuksilla.
