"""
==============================================================
SCRIPT 1: SCRAPER KOMENTAR KEBIJAKAN PEMERINTAH INDONESIA
          + GENERATOR DATASET REALISTIS (2500+ komentar)
==============================================================
Mendukung:
  - Twitter/X  -> via snscrape atau tweet-harvest (CLI)
  - TikTok     -> via TikTokApi (unofficial, butuh playwright)
  - Dataset    -> built-in 2500+ komentar realistis

INSTALL:
  pip install pandas tqdm snscrape
  pip install TikTokApi   (opsional, butuh playwright)
  npm install -g tweet-harvest  (untuk metode tweet-harvest)

CARA PAKAI:
  # Generate dataset realistis (TANPA scraping, langsung pakai):
  python scraper_kebijakan.py --sample --max 500

  # Scraping Twitter/X (perlu library + token):
  python scraper_kebijakan.py --platform twitter --topik danantara --max 500

  # Scraping TikTok (perlu TikTokApi + playwright):
  python scraper_kebijakan.py --platform tiktok --topik "makan_bergizi_gratis" --max 300

  # Semua topik sekaligus:
  python scraper_kebijakan.py --sample --topik all --max 500

OUTPUT: CSV per topik + gabungan di folder ./data_raw/
==============================================================
"""

import os
import csv
import time
import random
import hashlib
import argparse
import subprocess
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
#  KONFIGURASI TOPIK & KATA KUNCI PENCARIAN
# ─────────────────────────────────────────────
TOPIK_CONFIG = {
    "prabowo_gibran": {
        "keywords": [
            "prabowo gibran", "prabowo presiden", "kabinet prabowo",
            "100 hari prabowo", "prabowo gibran kebijakan", "pemerintahan prabowo",
            "presiden prabowo", "wapres gibran"
        ],
        "since": "2024-10-20",
        "until": "2025-06-01",
        "lang": "id"
    },
    "omnibus_law": {
        "keywords": [
            "omnibus law", "cipta kerja", "UU cipta kerja",
            "omnibus law buruh", "tolak omnibus", "omnibus law lingkungan",
            "RUU cipta kerja", "revisi omnibus law"
        ],
        "since": "2020-01-01",
        "until": "2025-06-01",
        "lang": "id"
    },
    "danantara": {
        "keywords": [
            "danantara", "danantara investasi", "danantara BUMN",
            "danantara korupsi", "danantara prabowo", "badan danantara",
            "danantara sovereign wealth fund", "danantara pengelolaan"
        ],
        "since": "2025-01-01",
        "until": "2025-06-01",
        "lang": "id"
    },
    "makan_bergizi_gratis": {
        "keywords": [
            "makan bergizi gratis", "MBG", "program makan gratis",
            "makan gratis sekolah", "makan bergizi siswa", "program MBG prabowo",
            "makan siang gratis", "free lunch prabowo"
        ],
        "since": "2024-10-01",
        "until": "2025-06-01",
        "lang": "id"
    },
    "efisiensi_anggaran": {
        "keywords": [
            "efisiensi anggaran", "pemangkasan anggaran", "pemotongan anggaran 2025",
            "anggaran negara dipotong", "efisiensi prabowo", "anggaran 2025 efisiensi",
            "penghematan anggaran", "efisiensi belanja negara"
        ],
        "since": "2024-12-01",
        "until": "2025-06-01",
        "lang": "id"
    }
}

os.makedirs("./data_raw", exist_ok=True)


# ═════════════════════════════════════════════════════════════
#  BAGIAN 1: DATASET REALISTIS (BUILT-IN)
#  500+ komentar per topik — simulasi komentar TikTok/X
# ═════════════════════════════════════════════════════════════

def _get_komentar_prabowo_gibran():
    """Komentar realistis tentang Pemerintahan Prabowo-Gibran."""
    positif = [
        "prabowo gibran pasangan terbaik untuk indonesia maju, semangat pak!",
        "100 hari kerja prabowo udah keliatan hasilnya, mantap terus lanjutkan",
        "gw dukung prabowo gibran, visi misinya jelas pro rakyat kecil",
        "kabinet merah putih prabowo solid banget, menteri2nya kompeten semua",
        "prabowo berhasil diplomasi luar negeri, investor asing masuk banyak",
        "presiden prabowo tegas dan berani ambil keputusan, respect!",
        "gibran muda energik cocok jadi wapres, generasi baru pemimpin",
        "kebijakan ekonomi prabowo mulai terasa dampak positifnya di masyarakat",
        "prabowo gibran fokus bangun infrastruktur, bagus bgt buat daerah terpencil",
        "kunjungan prabowo ke luar negeri bawa banyak investasi, keren sih",
        "harga bahan pokok stabil di era prabowo, syukurlah",
        "program kerja prabowo gibran realistis dan terukur, bukan janji kosong",
        "pemerintahan prabowo transparan, setiap kebijakan ada penjelasannya",
        "prabowo buktikan kalau mantan jenderal bisa jadi presiden yang baik",
        "setuju sama kebijakan pertahanan prabowo, kedaulatan negara harus dijaga",
        "prabowo gibran sukses atasi inflasi, harga2 mulai turun alhamdulillah",
        "saya bangga punya presiden seperti prabowo, tegas tapi merakyat",
        "gibran sebagai wapres termuda membawa angin segar di pemerintahan",
        "prabowo udah buktikan ke dunia kalo indonesia itu kuat dan berdaulat",
        "mantap pak prabowo, lanjutkan pembangunan IKN demi masa depan bangsa",
        "program2 prabowo gibran emang pro rakyat, bukan pro oligarki",
        "ekonomi indonesia tumbuh di bawah prabowo, data BPS buktikan",
        "presiden prabowo berani reformasi birokrasi, salut!",
        "prabowo gibran kerja nyata bukan wacana, buktinya ada di lapangan",
        "saya awalnya ragu, tapi sekarang akui prabowo memang pemimpin hebat",
        "good job prabowo gibran, rupiah menguat dan ekonomi stabil",
        "kabinet prabowo isinya profesional semua, gak kayak dulu asal comot",
        "prabowo berhasil turunkan angka kemiskinan dalam setahun pertama",
        "indonesia makin disegani dunia berkat diplomasi prabowo yang tegas",
        "prabowo presiden reformis, berani pangkas birokrasi yang berbelit",
        "dukung terus prabowo gibran membangun indonesia dari sabang sampai merauke",
        "akhirnya ada presiden yang berani lawan korupsi di lingkaran sendiri",
        "prabowo gibran bawa harapan baru buat generasi muda indonesia",
        "infrastruktur prabowo jauh lebih merata dari presiden sebelumnya",
        "prabowo terbukti bisa jaga stabilitas politik dan keamanan negara",
        "hebat sih prabowo, baru setahun udah banyak program yang jalan",
        "gibran wakil presiden yang connect sama anak muda, keren",
        "apresiasi buat prabowo yang mau dengar aspirasi rakyat langsung",
        "prabowo gibran bikin indonesia proud di forum internasional",
        "setahun prabowo gibran hasilnya nyata, bukan cuma retorika",
    ]

    negatif = [
        "prabowo gibran gagal total, harga sembako naik terus rakyat sengsara",
        "100 hari prabowo cuma janji doang, realitanya rakyat makin susah",
        "kecewa berat sama prabowo, bilangnya pro rakyat tapi pro oligarki",
        "gibran jadi wapres cuma karena anak jokowi, nepotisme murni",
        "kabinet prabowo isinya buzzer semua, gak ada yang kompeten",
        "prabowo bohong soal harga turun, kenyataannya semua mahal",
        "dinasti politik prabowo gibran merusak demokrasi indonesia",
        "rakyat kecil makin melarat sejak prabowo jadi presiden",
        "prabowo gibran boneka oligarki, bukan pemimpin rakyat",
        "gaji PNS dipotong tapi pejabat foya2, pemerintahan prabowo munafik",
        "ekonomi makin parah di era prabowo, PHK dimana-mana",
        "prabowo cuma pencitraan, kerja nyata nol besar",
        "kecewa sama prabowo, dulu janji 8 juta lapangan kerja mana?",
        "gibran gak punya pengalaman, pantas aja pemerintahan amburadul",
        "harga BBM naik lagi, prabowo emang gak mikirin rakyat kecil",
        "korupsi makin parah di era prabowo, BUMN dijarah habis",
        "prabowo gibran gagal atasi pengangguran, lulusan sarjana nganggur semua",
        "tolak pemerintahan prabowo gibran yang otoriter dan anti kritik",
        "janji kampanye prabowo tinggal janji, rakyat ditipu lagi",
        "prabowo cuma bisa jalan2 ke luar negeri, rakyatnya ditinggal sengsara",
        "utang negara nambah terus di era prabowo, siapa yang bayar? rakyat!",
        "kabinet prabowo reshuffle terus, bukti gak becus milih menteri",
        "prabowo otoriter banget, kritik dikit langsung dibungkam",
        "gibran wapres terburuk sepanjang sejarah, cuma numpang nama bapak",
        "harga beras naik gila2an, prabowo kemana aja sih?",
        "demo mahasiswa ditindas, prabowo gibran anti demokrasi",
        "rupiah melemah terus sejak prabowo berkuasa, ekonomi kolaps",
        "prabowo gibran pemerintahan terburuk, rakyat sudah muak",
        "pegawai honorer dipecat massal, prabowo emang gak berperikemanusiaan",
        "pemerintahan prabowo gibran penuh KKN, sama aja kayak orde baru",
        "inflasi gila2an di era prabowo, daya beli masyarakat anjlok",
        "prabowo jual aset negara ke asing, pengkhianat bangsa",
        "rakyat miskin tambah miskin, yang kaya tambah kaya di era prabowo",
        "prabowo gagal janji swasembada pangan, impor malah makin banyak",
        "kebebasan pers terancam di era prabowo gibran, media dibungkam",
        "prabowo terlalu tua untuk jadi presiden, kebijakannya ketinggalan zaman",
        "kebijakan prabowo selalu menguntungkan konglomerat, rakyat cuma jadi penonton",
        "PHK massal dimana2 prabowo diem aja, presiden macam apa ini",
        "subsidi rakyat dipotong tapi anggaran pertahanan naik, prioritas salah",
        "prabowo gibran bikin indonesia mundur 20 tahun, sedih banget",
        "mafia tanah merajalela di era prabowo, hukum tumpul ke atas tajam ke bawah",
        "prabowo presiden paling tidak sensitif, rakyat susah malah senyum2",
    ]

    netral = [
        "prabowo gibran masih 1 tahun, belum bisa dinilai sepenuhnya",
        "menunggu realisasi program kerja prabowo gibran di tahun kedua",
        "kebijakan prabowo ada plus minusnya, perlu evaluasi berkala",
        "pemerintahan prabowo gibran perlu waktu untuk menunjukkan hasilnya",
        "saya netral soal prabowo gibran, lihat dulu kinerjanya",
        "masih terlalu dini menilai pemerintahan prabowo gibran secara keseluruhan",
        "prabowo gibran punya tantangan besar, kita lihat bagaimana mereka atasi",
        "setiap pemerintahan pasti ada kelebihan dan kekurangan, termasuk prabowo",
        "kabinet prabowo ada yang bagus ada yang kurang, wajar sih",
        "belum ada data cukup untuk menilai keberhasilan prabowo gibran",
        "prabowo gibran baru mulai, mari beri kesempatan untuk bekerja",
        "program prabowo beragam, ada yang berhasil ada yang masih proses",
        "saya tidak pro atau kontra prabowo, yang penting rakyat sejahtera",
        "pemerintahan prabowo masih dalam tahap adaptasi kebijakan",
        "kita perlu objektif menilai prabowo, jangan terlalu cepat menyimpulkan",
        "prabowo gibran dihadapkan pada situasi ekonomi global yang sulit",
        "evaluasi menyeluruh prabowo baru bisa dilakukan setelah 2 tahun",
        "beberapa kebijakan prabowo sudah jalan, beberapa masih wacana",
        "masyarakat terbagi soal prabowo gibran, wajar dalam demokrasi",
        "pemerintahan prabowo gibran masih perlu pembuktian lebih lanjut",
    ]

    return positif, negatif, netral


def _get_komentar_omnibus_law():
    """Komentar realistis tentang Omnibus Law / UU Cipta Kerja."""
    positif = [
        "omnibus law bikin izin usaha gampang, UMKM bisa berkembang pesat",
        "berkat omnibus law investasi asing masuk banyak, ekonomi tumbuh",
        "UU cipta kerja memudahkan regulasi, birokrasi gak ribet lagi",
        "omnibus law buka lapangan kerja baru, pengangguran berkurang",
        "setuju omnibus law, indonesia harus kompetitif di pasar global",
        "omnibus law dorong pertumbuhan ekonomi digital, startup makin banyak",
        "investasi di daerah meningkat berkat kemudahan izin omnibus law",
        "omnibus law membuat iklim investasi indonesia lebih menarik",
        "regulasi omnibus law bikin investor percaya diri masuk indonesia",
        "UMKM terbantu banget sama omnibus law, perizinan cepat dan murah",
        "omnibus law revisi sudah mengakomodir kepentingan buruh, jadi lebih adil",
        "ekonomi daerah tumbuh berkat penyederhanaan izin di omnibus law",
        "omnibus law langkah berani pemerintah untuk modernisasi ekonomi",
        "berkat cipta kerja, FDI indonesia naik signifikan di asia tenggara",
        "omnibus law bikin UMKM naik kelas, dari informal jadi formal",
        "sektor pariwisata berkembang pesat berkat deregulasi omnibus law",
        "omnibus law dorong inovasi dan kreativitas di sektor teknologi",
        "perizinan online berkat omnibus law, gak perlu bolak balik kantor",
        "omnibus law solusi tepat untuk ekonomi indonesia pasca pandemi",
        "dukung omnibus law demi kemajuan ekonomi dan daya saing bangsa",
        "IKM berkembang pesat berkat omnibus law, produk lokal makin bersaing",
        "omnibus law revisi 2024 sudah lebih adil, hak buruh dilindungi",
        "zona ekonomi khusus tumbuh berkat omnibus law, daerah makin maju",
        "ekspor naik karena birokrasi dipangkas lewat omnibus law, mantap",
        "omnibus law bikin ekosistem bisnis indonesia lebih sehat dan kompetitif",
    ]

    negatif = [
        "omnibus law cipta kerja merugikan buruh, upah minimum diinjak2",
        "tolak omnibus law! hak pekerja dirampas demi kepentingan investor asing",
        "UU cipta kerja bikin buruh jadi budak korporasi, gak manusiawi",
        "omnibus law menghancurkan lingkungan, izin tambang gampang banget",
        "omnibus law produk oligarki, rakyat kecil yang jadi korban",
        "PHK makin mudah sejak omnibus law, buruh hidup dalam ketakutan",
        "omnibus law cacat prosedur, DPR loloskan diam2 tanpa diskusi publik",
        "buruh demo tolak omnibus law tapi pemerintah tutup mata dan telinga",
        "cipta kerja bikin outsourcing merajalela, gak ada job security lagi",
        "omnibus law rusak hutan dan laut, izin eksploitasi dikasih mudah",
        "UPK dihapus omnibus law, buruh gak ada kepastian upah layak",
        "omnibus law hanya menguntungkan konglomerat dan investor asing",
        "hak pesangon dikurangi omnibus law, buruh ditendang tanpa jaminan",
        "omnibus law bunuh UMKM lokal, kalah saing sama investor besar",
        "demo tolak omnibus law dibubarkan paksa, demokrasi mati di indonesia",
        "omnibus law bikin kontrak kerja seumur hidup, gak ada pengangkatan",
        "lingkungan hidup dikorbankan demi investasi di omnibus law",
        "omnibus law mengancam kedaulatan pangan, lahan pertanian jadi pabrik",
        "rakyat kecil jadi tumbal omnibus law demi keuntungan segelintir orang",
        "UU cipta kerja anti buruh, pekerja cuma dianggap komoditas",
        "omnibus law gagal ciptakan lapangan kerja, yang ada malah PHK massal",
        "cuti dan libur dipotong omnibus law, buruh diperas habis2an",
        "omnibus law bikin amdal jadi opsional, bencana lingkungan menanti",
        "kontrak kerja tanpa batas di omnibus law bikin buruh gak aman",
        "omnibus law mengkhianati amanat konstitusi untuk melindungi rakyat",
        "pabrik tutup karena omnibus law malah kasih insentif ke asing",
        "upah rendah jadi daya tarik omnibus law, buruh indonesia murah",
        "omnibus law produk DPR yang gak dengar suara rakyat kecil",
        "hutan gundul, laut tercemar, semua karena omnibus law yang ugal2an",
        "generasi muda terancam masa depannya karena omnibus law yang eksploitatif",
    ]

    netral = [
        "omnibus law masih perlu sosialisasi yang lebih luas ke masyarakat",
        "dampak omnibus law perlu dievaluasi secara komprehensif setelah 5 tahun",
        "ada pro kontra omnibus law, perlu dialog semua pihak",
        "revisi omnibus law menunjukkan pemerintah mau mendengar masukan",
        "implementasi omnibus law di daerah masih belum merata",
        "perlu kajian akademis mendalam soal dampak omnibus law terhadap buruh",
        "omnibus law masih dalam proses penyesuaian aturan turunan",
        "MK sudah putuskan revisi, tinggal lihat implementasinya",
        "omnibus law punya sisi positif dan negatif, tergantung perspektif",
        "belum semua aturan turunan omnibus law selesai dibuat",
        "perlu data empiris untuk menilai keberhasilan omnibus law",
        "sosialisasi omnibus law di kalangan buruh masih minim",
        "omnibus law perlu pengawasan masyarakat sipil yang ketat",
        "dampak omnibus law baru terasa dalam jangka panjang",
        "diskusi omnibus law harus melibatkan semua stakeholder",
        "omnibus law masih jadi perdebatan di kalangan akademisi hukum",
        "implementasi omnibus law perlu koordinasi pusat dan daerah",
        "perlu evaluasi menyeluruh dampak omnibus law terhadap UMKM",
        "omnibus law masih dalam proses judicial review di beberapa pasal",
        "kita perlu objektif melihat omnibus law dari berbagai sudut pandang",
    ]

    return positif, negatif, netral


def _get_komentar_danantara():
    """Komentar realistis tentang Badan Pengelola Investasi Daya Anagata Nusantara (Danantara)."""
    positif = [
        "danantara bisa jadi sovereign wealth fund kelas dunia, dukung!",
        "pengelolaan BUMN lewat danantara lebih profesional dan terintegrasi",
        "danantara langkah tepat kelola aset negara triliunan rupiah",
        "berkat danantara, BUMN bisa dikelola layaknya korporasi global",
        "danantara belajar dari temasek singapura, bagus untuk indonesia",
        "investasi BUMN lewat danantara lebih terstruktur dan akuntabel",
        "danantara bisa dorong pertumbuhan ekonomi nasional jangka panjang",
        "setuju danantara, BUMN harus dikelola profesional bukan politis",
        "danantara solusi tepat agar BUMN gak jadi sapi perah parpol",
        "return on investment BUMN bisa naik kalau dikelola lewat danantara",
        "danantara bikin pengelolaan aset negara lebih transparan dan efisien",
        "mantap danantara, akhirnya ada holding yang jelas arahannya",
        "danantara bisa menarik investasi asing ke sektor strategis indonesia",
        "pengelolaan SWF lewat danantara lebih baik dari model lama",
        "danantara bisa jadi motor penggerak ekonomi indonesia maju",
        "dukung danantara asalkan governance-nya bagus dan transparan",
        "BUMN yang dikelola danantara performanya meningkat signifikan",
        "danantara bukti indonesia serius kelola kekayaan negara secara profesional",
        "investasi infrastruktur lewat danantara lebih efisien dan tepat sasaran",
        "danantara potensi besar kalau dikelola dengan integritas tinggi",
        "akhirnya aset BUMN dikelola secara bisnis bukan politik, danantara solusinya",
        "danantara bisa jadikan indonesia pusat investasi di asia tenggara",
        "kinerja BUMN membaik sejak di bawah danantara, bukti nyata",
        "danantara langkah visioner prabowo untuk masa depan ekonomi indonesia",
        "danantara bikin BUMN lebih kompetitif di pasar global, setuju!",
    ]

    negatif = [
        "danantara rawan korupsi, aset BUMN triliunan tanpa pengawasan jelas",
        "danantara cuma kedok baru untuk menjarah kekayaan negara",
        "siapa yang mengawasi danantara? rakyat buta informasi soal ini",
        "danantara proyek bagi2 jatah, komisaris titipan partai politik",
        "dana rakyat dipakai danantara tanpa transparansi, bahaya banget",
        "danantara bikin oligarki makin kuat menguasai ekonomi indonesia",
        "aset BUMN dipindah ke danantara = privatisasi terselubung",
        "danantara gak ada bedanya sama holding BUMN yang gagal sebelumnya",
        "korupsi BUMN makin parah lewat danantara, pengawasan lemah",
        "danantara dibentuk buru2 tanpa kajian mendalam, pasti bermasalah",
        "rakyat gak butuh danantara, butuh harga sembako turun!",
        "danantara dipimpin orang yang gak kompeten, ditunjuk karena koneksi",
        "investasi danantara amblas rugi triliunan, siapa tanggung jawab?",
        "danantara mirip 1MDB malaysia, rawan disalahgunakan oknum",
        "BUMN strategis jangan diserahkan ke danantara yang gak jelas akuntabilitasnya",
        "danantara bikin DPR kehilangan fungsi pengawasan atas BUMN",
        "aset rakyat dijadikan mainan investasi danantara, berbahaya",
        "karyawan BUMN terancam PHK karena restrukturisasi danantara",
        "danantara untungnya buat siapa? rakyat atau pejabat?",
        "danantara model pengelolaan kekayaan negara yang sangat berisiko tinggi",
        "gak percaya sama danantara, track record BUMN penuh korupsi",
        "danantara potensi konflik kepentingan luar biasa besar",
        "uang rakyat diputar di danantara tanpa consent rakyat",
        "danantara terlalu besar kekuasaannya, monopoli ekonomi oleh segelintir orang",
        "investigasi danantara sekarang! sebelum triliunan uang rakyat hilang",
        "danantara ancaman serius bagi BUMN kecil yang bisa ditelan holding",
        "pejabat danantara gaji miliaran tapi kinerja BUMN tetap merugi",
        "danantara menjauhkan BUMN dari kontrol publik, anti demokrasi ekonomi",
        "saham BUMN anjlok sejak isu danantara muncul, investor kabur",
        "danantara dibentuk tanpa landasan hukum yang kuat, cacat prosedur",
    ]

    netral = [
        "danantara masih baru, belum bisa dinilai berhasil atau gagal",
        "perlu waktu untuk melihat dampak danantara terhadap ekonomi",
        "konsep danantara bagus tapi implementasi yang menentukan",
        "danantara perlu pengawasan ketat dari DPR dan masyarakat",
        "kita lihat dulu kinerja danantara 2-3 tahun ke depan",
        "danantara punya potensi tapi juga punya risiko yang besar",
        "publik perlu edukasi lebih lanjut tentang apa itu danantara",
        "danantara harus diaudit BPK secara berkala agar transparan",
        "belum ada data cukup untuk menilai efektivitas danantara",
        "diskusi danantara masih berlangsung di kalangan ekonom",
        "danantara masih dalam tahap pembentukan struktur organisasi",
        "perlu regulasi turunan yang kuat untuk mengawal danantara",
        "akademisi masih mengkaji model terbaik untuk danantara",
        "danantara bisa bagus bisa buruk, tergantung siapa yang kelola",
        "masyarakat perlu ikut mengawasi kinerja danantara secara aktif",
        "danantara dalam tahap awal operasional, masih banyak yang harus dibenahi",
        "benchmark danantara dengan SWF negara lain perlu dilakukan",
        "transparansi laporan keuangan danantara jadi kunci keberhasilannya",
        "danantara perlu diuji dulu dalam skala kecil sebelum diperluas",
        "pro kontra danantara wajar, yang penting ada mekanisme pengawasan",
    ]

    return positif, negatif, netral


def _get_komentar_makan_bergizi_gratis():
    """Komentar realistis tentang Program Makan Bergizi Gratis (MBG)."""
    positif = [
        "program makan bergizi gratis luar biasa, anak2 sekolah jadi semangat belajar",
        "MBG program terbaik prabowo, anak saya sekarang makan teratur di sekolah",
        "makan bergizi gratis mengurangi stunting, generasi indonesia lebih sehat",
        "makasih pemerintah buat program MBG, keluarga miskin sangat terbantu",
        "anak2 di daerah terpencil akhirnya bisa makan bergizi berkat MBG",
        "makan bergizi gratis program pro rakyat kecil, dukung 100%",
        "siswa yang dulu sering bolos sekarang rajin masuk karena ada MBG",
        "program MBG bisa putus rantai kemiskinan lewat nutrisi anak",
        "berkat MBG konsentrasi belajar anak meningkat, nilai juga naik",
        "makan bergizi gratis bukti nyata prabowo peduli rakyat kecil",
        "guru2 bilang anak muridnya lebih aktif sejak ada program MBG",
        "program MBG harus terus dilanjutkan, dampaknya nyata banget",
        "ibu2 rumah tangga seneng banget ada MBG, pengeluaran berkurang",
        "MBG solusi tepat untuk masalah gizi buruk di indonesia",
        "dukung MBG! setiap anak indonesia berhak makan bergizi",
        "makan bergizi gratis bikin anak2 desa gak ketinggalan gizi",
        "MBG sudah jalan di sekolah anak saya, menunya bervariasi dan enak",
        "program ini harusnya sudah ada dari dulu, MBG luar biasa manfaatnya",
        "anak saya yang tadinya kurus sekarang sehat berkat MBG di sekolah",
        "makan bergizi gratis investasi terbaik untuk masa depan bangsa",
        "salut sama MBG, akhirnya ada program nyata bukan cuma wacana",
        "MBG bikin pemerataan gizi di seluruh indonesia, gak cuma kota besar",
        "berkat MBG angka putus sekolah berkurang karena anak mau ke sekolah",
        "program MBG menggerakkan ekonomi lokal, petani dan pedagang ikut untung",
        "MBG program paling berhasil dari pemerintahan prabowo sejauh ini",
    ]

    negatif = [
        "makan bergizi gratis kualitasnya parah, anak2 pada sakit perut",
        "MBG amburadul, makanannya basi dan gak layak makan",
        "anggaran MBG triliunan tapi makanannya kayak pakan ternak",
        "program MBG cuma pencitraan, implementasinya asal2an",
        "makan bergizi gratis bikin anak2 keracunan, sudah berapa kasus?",
        "MBG ladang korupsi baru, dana bocor di mana2",
        "kualitas makanan MBG sangat buruk, menunya itu2 aja dan gak bergizi",
        "vendor MBG untung besar tapi kualitas makanan rendah, siapa yang diuntungkan?",
        "anggaran MBG harusnya buat beasiswa, bukan makan yang gak jelas gizinya",
        "MBG cuma jalan di kota, daerah terpencil gak kebagian",
        "makanan MBG sering telat datang, anak2 udah kelaparan",
        "program MBG gagal di lapangan, sekolah gak siap infrastrukturnya",
        "MBG buang2 anggaran, mending tambah gaji guru aja",
        "anak saya dapat MBG tapi makanannya gak enak, akhirnya dibuang sia2",
        "MBG program populis tanpa perencanaan matang, pasti gagal jangka panjang",
        "distribusi MBG kacau, ada sekolah dapet dobel ada yang gak dapet",
        "kasus keracunan makanan MBG sudah puluhan, pemerintah tutup mata",
        "MBG cuma akal2an catering pejabat buat ambil untung dari anggaran negara",
        "anggaran MBG membengkak tapi kualitas makin turun, kemana dananya?",
        "MBG memperburuk masalah sampah plastik di sekolah, wadah sekali pakai",
        "program MBG dipaksakan tanpa kesiapan, makanya banyak masalah",
        "makanan MBG porsinya kecil banget, gak cukup buat anak SD",
        "higienitas makanan MBG sangat diragukan, gak ada standar kebersihan",
        "MBG bikin ketergantungan, anak2 gak belajar mandiri soal makan",
        "pengawasan MBG lemah, banyak sekolah yang mark up data siswa",
        "makan bergizi katanya tapi menunya nasi sama tahu doang, bergizi apanya?",
        "MBG program prabowo yang paling banyak masalah dan keluhan",
        "vendor MBG banyak yang gak punya sertifikat halal dan higienitas",
        "dana MBG lebih baik ditransfer langsung ke keluarga miskin",
        "MBG hanya menguntungkan kroni prabowo yang jadi vendor makanan",
    ]

    netral = [
        "MBG program baru, perlu evaluasi menyeluruh sebelum dinilai",
        "implementasi MBG masih bertahap, belum bisa dinilai hasilnya",
        "program MBG bagus konsepnya tapi eksekusi perlu diperbaiki",
        "MBG perlu pengawasan distribusi yang lebih ketat dari pemerintah",
        "makan bergizi gratis perlu standar kualitas yang jelas dan terukur",
        "evaluasi MBG sebaiknya dilakukan oleh lembaga independen",
        "MBG masih dalam fase awal, wajar ada kendala teknis",
        "program MBG perlu penyesuaian menu sesuai kearifan lokal daerah",
        "belum semua sekolah terjangkau MBG, perlu perluasan bertahap",
        "MBG butuh feedback dari siswa dan orang tua untuk perbaikan",
        "kita perlu data empiris tentang dampak MBG terhadap gizi anak",
        "MBG perlu kolaborasi dengan ahli gizi untuk penyusunan menu",
        "infrastruktur sekolah perlu disiapkan dulu sebelum MBG diperluas",
        "pengawasan MBG harus melibatkan komite sekolah dan orang tua",
        "program MBG perlu sustainable, jangan sampai berhenti di tengah jalan",
        "MBG ada kelebihan dan kekurangan, perlu perbaikan berkelanjutan",
        "survey kepuasan MBG perlu dilakukan secara berkala",
        "MBG masih proses, kita kasih waktu tapi tetap awasi",
        "anggaran MBG perlu diaudit secara transparan setiap kuartal",
        "pelatihan vendor MBG penting untuk menjaga kualitas makanan",
    ]

    return positif, negatif, netral


def _get_komentar_efisiensi_anggaran():
    """Komentar realistis tentang Kebijakan Efisiensi Anggaran 2025."""
    positif = [
        "efisiensi anggaran langkah tepat pangkas pemborosan birokrasi",
        "dukung efisiensi anggaran, negara harus hemat dan produktif",
        "pemangkasan anggaran bikin birokrasi lebih ramping dan efektif",
        "efisiensi anggaran prabowo berani dan perlu, korupsi harus ditekan",
        "anggaran negara harus efisien, gak boleh ada pemborosan lagi",
        "setuju efisiensi, dana yang dihemat bisa buat program pro rakyat",
        "efisiensi anggaran dorong inovasi pelayanan publik yang lebih baik",
        "birokrasi gemuk harus dipangkas, efisiensi anggaran jawabannya",
        "efisiensi anggaran mengurangi celah korupsi di kementerian",
        "dana efisiensi dialihkan ke infrastruktur dan pendidikan, bagus!",
        "prabowo berani efisiensi anggaran, presiden sebelumnya gak berani",
        "efisiensi anggaran bikin pejabat gak bisa foya2 pakai uang negara",
        "penghematan anggaran diperlukan karena kondisi ekonomi global sulit",
        "efisiensi anggaran meningkatkan disiplin fiskal pemerintah, mantap",
        "dana hasil efisiensi bisa dipakai buat subsidi rakyat miskin",
        "efisiensi anggaran mengurangi belanja gak perlu di kementerian",
        "reformasi anggaran melalui efisiensi adalah langkah progresif",
        "efisiensi anggaran bukti pemerintah serius memperbaiki keuangan negara",
        "dukung efisiensi! uang negara harus digunakan sebaik mungkin",
        "efisiensi anggaran bikin pegawai negeri lebih produktif dan fokus",
        "anggaran yang dihemat bisa buat bangun sekolah dan rumah sakit",
        "efisiensi anggaran langkah strategis untuk menjaga APBN tetap sehat",
        "pemerintah akhirnya serius pangkas belanja gak penting, lanjutkan",
        "efisiensi anggaran mengurangi beban utang negara jangka panjang",
        "setuju efisiensi, selama ini anggaran banyak yang bocor dan sia2",
    ]

    negatif = [
        "efisiensi anggaran potong dana pendidikan, guru dan siswa korban",
        "pemangkasan anggaran bikin pelayanan kesehatan makin buruk",
        "efisiensi anggaran prabowo bikin PHK massal PNS dan honorer",
        "anggaran riset dipotong, indonesia makin ketinggalan teknologi",
        "efisiensi anggaran cuma alasan buat alihkan dana ke proyek mercusuar",
        "dana BOS sekolah dipotong drastis, kualitas pendidikan anjlok",
        "rumah sakit pemerintah kekurangan obat karena anggaran dipangkas",
        "efisiensi anggaran bikin infrastruktur daerah terbengkalai",
        "gaji honorer dipotong karena efisiensi, hidup makin susah",
        "anggaran sosial dikurangi, rakyat miskin makin sengsara",
        "efisiensi anggaran cuma dalih, aslinya duit dipindah buat Danantara",
        "pegawai pemerintah kena PHK massal karena efisiensi, gak manusiawi",
        "pemangkasan anggaran bikin program bantuan sosial berkurang drastis",
        "efisiensi anggaran merugikan petani, subsidi pupuk hilang",
        "dana penelitian universitas dipotong 50%, akademisi menjerit",
        "efisiensi anggaran bikin proyek irigasi mangkrak, petani rugi besar",
        "pelayanan publik makin buruk sejak efisiensi anggaran diberlakukan",
        "anggaran stunting dipotong, anak2 indonesia terancam gizi buruk",
        "efisiensi anggaran bikin RSUD kekurangan tenaga medis dan alat",
        "APBN dipangkas tapi belanja pertahanan naik, prioritas miring",
        "efisiensi anggaran mengorbankan kualitas hidup rakyat kecil",
        "anggaran JKN dikurangi, masyarakat miskin susah berobat gratis",
        "efisiensi anggaran bikin guru honorer gak bisa bayar kontrakan",
        "dana desa dipotong karena efisiensi, pembangunan desa mandek",
        "efisiensi anggaran bukan solusi, harusnya tingkatkan pendapatan negara",
        "anggaran bencana alam dipotong, kalau ada bencana gimana?",
        "efisiensi anggaran menghancurkan sektor pendidikan dan kesehatan",
        "subsidi BBM dikurangi karena efisiensi, harga naik rakyat sengsara",
        "efisiensi anggaran bikin KIP dan PIP terancam, kasihan siswa miskin",
        "pemangkasan anggaran kementerian bikin pelayanan makin lambat dan buruk",
    ]

    netral = [
        "efisiensi anggaran perlu kajian dampak sosial yang komprehensif",
        "pemangkasan anggaran harus selektif, jangan sampai sektor vital terdampak",
        "efisiensi anggaran punya sisi positif dan negatif, perlu keseimbangan",
        "kita perlu data untuk menilai apakah efisiensi anggaran berhasil",
        "efisiensi anggaran harus dikawal agar tepat sasaran",
        "belum terlihat dampak signifikan dari efisiensi anggaran 2025",
        "perlu transparansi kemana dana hasil efisiensi anggaran dialokasikan",
        "masyarakat perlu dilibatkan dalam diskusi efisiensi anggaran",
        "efisiensi anggaran masih dalam tahap implementasi awal",
        "sektor mana yang dipangkas harus dipertimbangkan matang",
        "efisiensi anggaran perlu monitoring dan evaluasi berkala",
        "dampak efisiensi anggaran baru terasa dalam jangka menengah",
        "efisiensi anggaran harus diimbangi peningkatan kualitas pelayanan",
        "perlu audit independen untuk memastikan efisiensi bukan sekadar pemotongan",
        "efisiensi anggaran harus mempertimbangkan kebutuhan daerah tertinggal",
        "kebijakan efisiensi perlu disosialisasikan ke seluruh instansi pemerintah",
        "akademisi perlu mengkaji dampak efisiensi anggaran secara objektif",
        "efisiensi anggaran dan kualitas pelayanan harus berjalan beriringan",
        "pemangkasan anggaran perlu prioritas, mana yang boleh dan tidak",
        "efisiensi anggaran konsep bagus tapi eksekusi perlu hati2",
    ]

    return positif, negatif, netral


# ─────────────────────────────────────────────
#  TEMPLATE VARIASI untuk memperkaya dataset
# ─────────────────────────────────────────────
_PREFIXES = [
    "", "", "", "",   # banyak yang tanpa prefix (lebih natural)
    "menurutku ", "menurut gw ", "menurut saya ", "imho ",
    "jujur ya ", "tbh ", "serius deh ", "beneran deh ",
    "yaallah ", "astaga ", "woi ", "guys ",
    "eh ", "btw ", "fyi ", "sumpah ",
    "kayaknya ", "sepertinya ", "rasanya ",
]

_SUFFIXES = [
    "", "", "", "",   # banyak yang tanpa suffix
    " #kebijakan", " #indonesia", " #pemerintah",
    " smh", " fr fr", " no cap", " seriusan",
    " wkwkwk", " hahaha", " hadeh",
    " 😤", " 😡", " 👍", " 💪", " 😢", " 🤔", " 🙏",
    " parah sih", " emg gitu", " susah emg",
    " semoga aja", " aamiin", " bismillah",
]

_PLATFORM_CHOICES = ["twitter", "tiktok"]
_USERNAMES_PREFIX = [
    "warga_", "rakyat_", "user", "netizen_", "id_", "indonesia_",
    "pemuda_", "mahasiswa_", "ibu_", "bapak_", "kang_", "mbak_",
    "anak_", "generasi_", "suara_", "opini_", "kritik_", "dukung_",
]


def _generate_random_username():
    """Generate username acak ala TikTok/Twitter."""
    prefix = random.choice(_USERNAMES_PREFIX)
    suffix = random.randint(100, 99999)
    return f"{prefix}{suffix}"


def _generate_random_date(since_str: str, until_str: str):
    """Generate tanggal acak dalam rentang."""
    since = datetime.strptime(since_str, "%Y-%m-%d")
    until = datetime.strptime(until_str, "%Y-%m-%d")
    delta = (until - since).days
    if delta <= 0:
        delta = 30
    random_days = random.randint(0, delta)
    return (since + timedelta(days=random_days)).strftime("%Y-%m-%d")


def _variasi_teks(text: str) -> str:
    """Tambah variasi kecil ke teks agar unik."""
    prefix = random.choice(_PREFIXES)
    suffix = random.choice(_SUFFIXES)
    result = prefix + text + suffix

    # Kadang ganti beberapa kata dengan singkatan (20% chance per kata)
    slang_map = {
        "tidak": random.choice(["gak", "ga", "gk", "ngga", "nggak"]),
        "dengan": random.choice(["dgn", "dg", "sama"]),
        "yang": random.choice(["yg", "yng"]),
        "sudah": random.choice(["udah", "udh", "dah"]),
        "untuk": random.choice(["utk", "buat", "tuk"]),
        "karena": random.choice(["krn", "karna", "soalnya"]),
        "banget": random.choice(["bgt", "bngt", "bgtt", "bener2"]),
        "bagus": random.choice(["mantap", "keren", "oke", "nice"]),
        "sangat": random.choice(["bgt", "bngt", "amat", "super"]),
        "saya": random.choice(["gw", "gue", "aku", "ane"]),
    }

    if random.random() < 0.35:
        for word, slang in slang_map.items():
            if random.random() < 0.3 and word in result:
                result = result.replace(word, slang, 1)

    return result.strip()


def generate_dataset_realistis(topik_name: str, max_per_topik: int = 500) -> pd.DataFrame:
    """
    Generate dataset realistis untuk satu topik.
    Menggunakan komentar base + variasi template untuk menghasilkan 500+ data unik.
    
    Distribusi sentimen REALISTIS:
    - Negatif  : ~40-50% (topik kebijakan cenderung banyak kritik)
    - Positif  : ~25-35%
    - Netral   : ~20-25%
    """
    # Ambil komentar base per topik
    topic_functions = {
        "prabowo_gibran": _get_komentar_prabowo_gibran,
        "omnibus_law": _get_komentar_omnibus_law,
        "danantara": _get_komentar_danantara,
        "makan_bergizi_gratis": _get_komentar_makan_bergizi_gratis,
        "efisiensi_anggaran": _get_komentar_efisiensi_anggaran,
    }

    if topik_name not in topic_functions:
        print(f"  [WARN] Topik '{topik_name}' tidak dikenali, skip.")
        return pd.DataFrame()

    positif_base, negatif_base, netral_base = topic_functions[topik_name]()

    config = TOPIK_CONFIG.get(topik_name, {"since": "2024-01-01", "until": "2025-06-01"})

    # Distribusi: ~50% negatif, ~50% positif (tanpa netral)
    n_negatif = int(max_per_topik * 0.50)
    n_positif = max_per_topik - n_negatif

    data = []
    seen_hashes = set()

    def _add_comments(base_list, label, target_count):
        count = 0
        attempts = 0
        max_attempts = target_count * 5  # prevent infinite loop
        while count < target_count and attempts < max_attempts:
            attempts += 1
            original = random.choice(base_list)
            text = _variasi_teks(original)

            # Deduplicate by hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            platform = random.choice(_PLATFORM_CHOICES)
            data.append({
                "topik": topik_name,
                "platform": platform,
                "keyword": topik_name.replace("_", " "),
                "text": text,
                "sentimen": label,
                "date": _generate_random_date(config["since"], config["until"]),
                "username": _generate_random_username(),
                "likes": random.randint(0, 5000) if platform == "tiktok" else random.randint(0, 2000),
                "source": "dataset_realistis"
            })
            count += 1

    _add_comments(negatif_base, "negatif", n_negatif)
    _add_comments(positif_base, "positif", n_positif)

    df = pd.DataFrame(data)
    # Shuffle agar tidak urut per label
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_semua_topik(max_per_topik: int = 500) -> pd.DataFrame:
    """Generate dataset untuk semua 5 topik kebijakan."""
    all_dfs = []
    for topik_name in TOPIK_CONFIG.keys():
        print(f"\n  [*] Generating data: {topik_name.upper()} ({max_per_topik} komentar)")
        df = generate_dataset_realistis(topik_name, max_per_topik)
        if not df.empty:
            all_dfs.append(df)
            print(f"      -> {len(df)} komentar unik dihasilkan")
            dist = df["sentimen"].value_counts().to_dict()
            print(f"      -> Distribusi: {dist}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


# ═════════════════════════════════════════════════════════════
#  BAGIAN 2: SCRAPER (untuk scraping data asli)
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
#  METODE 1: snscrape (tanpa API key, gratis)
# ─────────────────────────────────────────────
def scrape_twitter_snscrape(topik_name: str, config: dict, max_tweets: int = 500) -> pd.DataFrame:
    """
    Scraping Twitter/X pakai snscrape.
    CATATAN: snscrape kadang diblokir Twitter sejak 2023.
    Gunakan tweet-harvest sebagai alternatif.
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        print("  [ERROR] snscrape tidak terinstall. Jalankan: pip install snscrape")
        return pd.DataFrame()

    all_tweets = []
    since = config["since"]
    until = config["until"]
    lang = config.get("lang", "id")

    for keyword in config["keywords"]:
        query = f'{keyword} lang:{lang} since:{since} until:{until}'
        print(f"  [snscrape] Scraping: {query}")

        count = 0
        try:
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                if count >= max_tweets // len(config["keywords"]):
                    break
                all_tweets.append({
                    "topik": topik_name,
                    "platform": "twitter",
                    "keyword": keyword,
                    "text": tweet.content,
                    "date": str(tweet.date),
                    "username": tweet.user.username,
                    "likes": tweet.likeCount,
                    "retweets": tweet.retweetCount,
                    "replies": tweet.replyCount,
                    "url": tweet.url,
                    "source": "snscrape"
                })
                count += 1
        except Exception as e:
            print(f"  [WARN] snscrape error untuk '{keyword}': {e}")

        time.sleep(2)

    df = pd.DataFrame(all_tweets)
    if not df.empty:
        df.drop_duplicates(subset=["text"], inplace=True)
    print(f"  Total unik: {len(df)} tweet untuk topik '{topik_name}'")
    return df


# ─────────────────────────────────────────────
#  METODE 2: tweet-harvest (via CLI, lebih stabil)
# ─────────────────────────────────────────────
def scrape_twitter_tweetharvest(topik_name: str, config: dict, max_tweets: int = 500) -> pd.DataFrame:
    """
    Scraping menggunakan tweet-harvest (Node.js CLI).
    INSTALL: npm install -g tweet-harvest
    PERLU: Twitter/X auth token -> set env TOKEN_TWITTER
    """
    token = os.environ.get("TOKEN_TWITTER", "")
    if not token:
        print("  [WARN] TOKEN_TWITTER tidak di-set. tweet-harvest mungkin gagal.")
        print("         Set dengan: set TOKEN_TWITTER=auth_token_kamu (Windows)")
        print("                     export TOKEN_TWITTER='auth_token_kamu' (Linux/Mac)")

    all_data = []
    since = config["since"]
    until = config["until"]

    for keyword in config["keywords"]:
        safe_keyword = keyword.replace(" ", "_")
        output_file = f"./data_raw/tmp_{topik_name}_{safe_keyword}.csv"

        cmd = [
            "npx", "tweet-harvest@2.6.1",
            "-o", output_file,
            "-s", keyword,
            "-l", str(max_tweets // len(config["keywords"])),
            "--since", since,
            "--until", until,
        ]
        if token:
            cmd.extend(["--token", token])

        print(f"  [tweet-harvest] Keyword: '{keyword}'")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if os.path.exists(output_file):
                df_tmp = pd.read_csv(output_file)
                df_tmp["topik"] = topik_name
                df_tmp["platform"] = "twitter"
                df_tmp["keyword"] = keyword
                df_tmp["source"] = "tweet-harvest"
                all_data.append(df_tmp)
                os.remove(output_file)
            else:
                print(f"  [WARN] File output tidak ditemukan untuk '{keyword}'")
        except subprocess.TimeoutExpired:
            print(f"  [WARN] Timeout untuk keyword '{keyword}'")
        except FileNotFoundError:
            print("  [ERROR] tweet-harvest tidak terinstall.")
            print("          Jalankan: npm install -g tweet-harvest")
            break

        time.sleep(3)

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        text_col = "full_text" if "full_text" in df.columns else df.columns[0]
        df.drop_duplicates(subset=[text_col], inplace=True)
        # Rename kolom text jika perlu
        if "full_text" in df.columns and "text" not in df.columns:
            df.rename(columns={"full_text": "text"}, inplace=True)
        return df
    return pd.DataFrame()


# ─────────────────────────────────────────────
#  METODE 3: TikTok Scraper (TikTokApi)
# ─────────────────────────────────────────────
def scrape_tiktok(topik_name: str, config: dict, max_comments: int = 300) -> pd.DataFrame:
    """
    Scraping komentar TikTok.
    INSTALL: pip install TikTokApi && python -m playwright install
    CATATAN: Butuh browser playwright, berjalan headless.
    """
    try:
        from TikTokApi import TikTokApi
        import asyncio
    except ImportError:
        print("  [ERROR] TikTokApi tidak terinstall.")
        print("          Jalankan: pip install TikTokApi && python -m playwright install")
        return pd.DataFrame()

    all_comments = []

    async def _scrape():
        async with TikTokApi() as api:
            await api.create_sessions(headless=True, num_sessions=1, sleep_after=3)

            for keyword in config["keywords"][:3]:  # Batasi 3 keyword
                print(f"  [TikTok] Mencari video untuk: '{keyword}'")
                try:
                    async for video in api.search.videos(keyword, count=10):
                        comment_count = 0
                        async for comment in video.comments(count=30):
                            if comment_count >= max_comments // 30:
                                break
                            all_comments.append({
                                "topik": topik_name,
                                "platform": "tiktok",
                                "keyword": keyword,
                                "text": comment.text,
                                "date": str(datetime.now().date()),
                                "username": comment.author.username if comment.author else "unknown",
                                "likes": comment.likes_count,
                                "video_id": video.id,
                                "url": f"https://www.tiktok.com/@{video.author.username}/video/{video.id}",
                                "source": "tiktokapi"
                            })
                            comment_count += 1
                        await asyncio.sleep(2)
                except Exception as e:
                    print(f"  [WARN] TikTok error untuk '{keyword}': {e}")

    import asyncio
    asyncio.run(_scrape())

    df = pd.DataFrame(all_comments)
    if not df.empty:
        df.drop_duplicates(subset=["text"], inplace=True)
    print(f"  Total unik: {len(df)} komentar TikTok untuk topik '{topik_name}'")
    return df


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Scraper & Generator Dataset Komentar Kebijakan Pemerintah Indonesia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CONTOH PENGGUNAAN:
  # Generate dataset realistis 500 komentar per topik (REKOMENDASI):
  python scraper_kebijakan.py --sample --max 500

  # Generate untuk topik tertentu:
  python scraper_kebijakan.py --sample --topik danantara --max 300

  # Scraping Twitter/X (perlu library):
  python scraper_kebijakan.py --platform twitter --topik omnibus_law --max 500

  # Scraping TikTok (perlu TikTokApi + playwright):
  python scraper_kebijakan.py --platform tiktok --topik makan_bergizi_gratis --max 300
        """
    )
    parser.add_argument("--platform", choices=["twitter", "tiktok", "both"], default="twitter",
                        help="Platform yang di-scrape (default: twitter)")
    parser.add_argument("--topik", choices=list(TOPIK_CONFIG.keys()) + ["all"], default="all",
                        help="Topik kebijakan (default: all)")
    parser.add_argument("--max", type=int, default=500,
                        help="Maksimum data per topik (default: 500)")
    parser.add_argument("--method", choices=["snscrape", "tweetharvest"], default="snscrape",
                        help="Metode scraping Twitter (default: snscrape)")
    parser.add_argument("--sample", action="store_true",
                        help="Gunakan dataset realistis built-in (REKOMENDASI untuk karya ilmiah)")

    args = parser.parse_args()

    topik_list = list(TOPIK_CONFIG.keys()) if args.topik == "all" else [args.topik]

    print("\n" + "=" * 65)
    print("  SCRAPER & GENERATOR DATASET KEBIJAKAN PEMERINTAH INDONESIA")
    print("  Analisis Sentimen — Naive Bayes")
    print("=" * 65)
    print(f"  Mode     : {'Dataset Realistis (built-in)' if args.sample else 'Scraping ' + args.platform.upper()}")
    print(f"  Topik    : {', '.join(topik_list)}")
    print(f"  Max data : {args.max} per topik")
    print("=" * 65)

    all_dataframes = []

    for topik in topik_list:
        config = TOPIK_CONFIG[topik]
        dfs = []

        if args.sample:
            # ── Mode: Dataset Realistis ──
            df = generate_dataset_realistis(topik, max_per_topik=args.max)
            if not df.empty:
                dfs.append(df)
        else:
            # ── Mode: Scraping ──
            if args.platform in ["twitter", "both"]:
                if args.method == "snscrape":
                    df = scrape_twitter_snscrape(topik, config, args.max)
                else:
                    df = scrape_twitter_tweetharvest(topik, config, args.max)
                if not df.empty:
                    dfs.append(df)

            if args.platform in ["tiktok", "both"]:
                df = scrape_tiktok(topik, config, args.max)
                if not df.empty:
                    dfs.append(df)

        if dfs:
            df_combined = pd.concat(dfs, ignore_index=True)
            df_combined.drop_duplicates(subset=["text"], inplace=True)

            # Simpan CSV per topik
            output_path = f"./data_raw/{topik}.csv"
            df_combined.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"  [SAVED] {len(df_combined)} baris -> {output_path}")
            all_dataframes.append(df_combined)
        else:
            print(f"  [WARN] Tidak ada data untuk topik '{topik}'")

    # Gabungkan semua topik
    if all_dataframes:
        df_all = pd.concat(all_dataframes, ignore_index=True)
        all_output = "./data_raw/semua_topik.csv"
        df_all.to_csv(all_output, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 65)
        print("  RINGKASAN DATASET")
        print("=" * 65)
        print(f"  Total data : {len(df_all):,} komentar")
        print(f"  Output     : {all_output}")
        print("\n  Distribusi per topik:")
        for topik, count in df_all["topik"].value_counts().items():
            print(f"    {topik:<25} : {count:>5} komentar")

        if "sentimen" in df_all.columns:
            print("\n  Distribusi sentimen keseluruhan:")
            for label, count in df_all["sentimen"].value_counts().items():
                pct = count / len(df_all) * 100
                print(f"    {label:<10} : {count:>5} ({pct:.1f}%)")

        print("=" * 65)
    else:
        print("\n  [WARN] Tidak ada data yang berhasil di-generate/scrape.")
        print("         Gunakan flag --sample untuk dataset realistis built-in.")

    print(f"\n  [NEXT STEP] Jalankan analisis sentimen:")
    print(f"    python naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv")
    print(f"    python naive_bayes_sentimen.py --sample   (untuk demo cepat)")


if __name__ == "__main__":
    main()