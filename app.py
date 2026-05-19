import os
import json
import re
import time
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from google.generativeai.types import HarmBlockThreshold, HarmCategory


def gemini_stream_chunks(response_stream):
    for chunk in response_stream:
        chunk_text = getattr(chunk, "text", "")
        if chunk_text:
            for token in re.findall(r"\S+\s*|\n", chunk_text):
                yield token


def split_output_sections(full_text):
    lines = full_text.splitlines()
    main_lines = []
    extra_lines = []
    in_extra_section = False

    for line in lines:
        if "🛠️" in line or "🎨" in line:
            in_extra_section = True
        if in_extra_section:
            extra_lines.append(line)
        else:
            main_lines.append(line)

    main_content = "\n".join(main_lines).strip()
    extra_content = "\n".join(extra_lines).strip()
    return main_content, extra_content


def render_output_details(full_text, download_key, show_main_content=True):
    main_content, extra_content = split_output_sections(full_text)

    if show_main_content and main_content:
        st.markdown(main_content)

    if extra_content:
        with st.expander("🛠️ Araç Çantasını ve Promptları Gör"):
            st.markdown(extra_content)

    st.divider()
    st.success("İçeriğiniz hazır. Tam çıktıyı aşağıdaki butonla indirebilirsiniz.")
    st.download_button(
        "📥 Bu Fikri İndir",
        full_text,
        file_name="icerik_fikri.txt",
        use_container_width=True,
        key=download_key,
    )


def get_default_api_key():
    api_key_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY")

    for key_name in api_key_names:
        try:
            secret_value = st.secrets[key_name]
        except Exception:
            secret_value = None

        if secret_value:
            return str(secret_value).strip()

    for key_name in api_key_names:
        env_value = os.getenv(key_name, "").strip()
        if env_value:
            return env_value

    env_path = Path(".env")
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key_name, value = line.split("=", 1)
                cleaned_key = key_name.strip()
                cleaned_value = value.strip().strip('"').strip("'")

                if cleaned_key in api_key_names and cleaned_value:
                    return cleaned_value
        except OSError:
            return None

    return None


def is_rate_limit_error(error):
    error_message = str(error).lower()
    rate_limit_indicators = (
        "429",
        "resourceexhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "too many requests",
    )
    return any(indicator in error_message for indicator in rate_limit_indicators)


def generate_content_with_retry(
    model,
    prompt,
    safety_settings,
    main_output_placeholder,
    max_attempts=3,
    show_spinner=True,
):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        spinner_message = (
            "Yapay zeka içerik stüdyosunda senaryonuzu kurguluyor... 🚀"
            if attempt == 1
            else "Sunucular yoğun, güvenli bir hat aranıyor... Lütfen bekleyin ⏳"
        )

        output = ""

        try:
            main_output_placeholder.empty()
            if attempt > 1:
                time.sleep(3)

            if show_spinner:
                with st.spinner(spinner_message):
                    response_stream = model.generate_content(
                        prompt,
                        safety_settings=safety_settings,
                        stream=True,
                    )

                    for token in gemini_stream_chunks(response_stream):
                        output += token
                        main_content, _ = split_output_sections(output)
                        display_content = main_content if main_content else "İçerik hazırlanıyor..."
                        main_output_placeholder.markdown(display_content + "▌")
            else:
                response_stream = model.generate_content(
                    prompt,
                    safety_settings=safety_settings,
                    stream=True,
                )

                for token in gemini_stream_chunks(response_stream):
                    output += token
                    main_content, _ = split_output_sections(output)
                    display_content = main_content if main_content else "İçerik hazırlanıyor..."
                    main_output_placeholder.markdown(display_content + "▌")

            return output

        except Exception as error:
            last_error = error
            if not is_rate_limit_error(error) or attempt == max_attempts:
                break

    raise last_error


def apply_mobile_first_styles():
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1080px;
                padding-top: 1rem;
                padding-right: 1rem;
                padding-bottom: 2rem;
                padding-left: 1rem;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid #e8ebf2;
            }

            [data-testid="stSidebar"] [role="radiogroup"] {
                gap: 0.45rem;
            }

            [data-testid="stSidebar"] [data-baseweb="radio"] label {
                border: 1px solid #dce3f3;
                border-radius: 12px;
                padding: 0.65rem 0.75rem;
                background: #f9fbff;
            }

            div.stButton > button,
            div.stDownloadButton > button {
                min-height: 3.2rem;
                font-weight: 600;
            }

            @media (max-width: 640px) {
                .block-container {
                    padding-right: 0.85rem;
                    padding-left: 0.85rem;
                }

                /* Mobilde sütunları alt alta yığ */
                [data-testid="stHorizontalBlock"] {
                    flex-wrap: wrap;
                }

                [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
                    width: 100% !important;
                    flex: 1 1 100% !important;
                    min-width: 100% !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_platform_specific_prompt(platform):
    if platform == "TikTok (Kısa Video)":
        return """
        TikTok için hızlı tüketilen, yüksek tempolu, maksimum 60 saniyelik bir kısa video içeriği üret.
        Tüm çıktı Türkçe olmalıdır.
        Video dik formatta (9:16) planlanmalıdır.
        İlk 3 saniye çok çarpıcı bir kanca ile başlamalıdır.
        Sahne geçişleri hızlı, dinamik ve kısa dikkat süresine uygun olmalıdır.
        Çıktı formatı:
        1. 🎣 **0-3 Saniye: Kanca:** İzleyiciyi anında videoda tutacak güçlü, merak uyandırıcı açılış.
        2. ⚡ **3-15 Saniye: Giriş:** Konunun hızlı ve net kurulumu.
        3. 🔥 **15-45 Saniye: Asıl İçerik:** En değerli bilgi, gösterim veya anlatımın yüksek tempolu akışı.
        4. 🚀 **45-60 Saniye: Kapanış ve Takip Çağrısı:** İzleyiciyi takibe, yoruma veya paylaşmaya yönlendiren kısa kapanış.
        5. 🛠️ **Üretim Çantası (Önerilen Araçlar):** Kullanıcıların bu içeriği üretmek için kullanabileceği araçlar:
           - Kurgu: CapCut, Premiere Pro
           - AI Seslendirme: ElevenLabs, CapCut Voice vb.
           - AI Video/Görsel: Runway, Veo, Midjourney, DALL-E vb.
        6. 🎨 **Görsel/Video AI İçin Hazır Prompt:** Yalnızca Türkçe, dik formata uygun (9:16) görsel ve video üretim promptları yaz.
           - Asla İngilizce prompt yazma.
           - Her bölüm için ayrı ayrı prompt ver; ASLA tüm akışı tek bir birleşik promptta toplama.
           - Hızlı geçişleri, mobil ekran uyumunu, yakın planları ve yüksek dikkat çekiciliği promptlara dahil et.
           - Şu formatı kullan:
             - Kanca Sahnesi Video Promptu: [Türkçe, 9:16 dik format, çok çarpıcı açılış, hızlı kamera hareketi, güçlü ışık ve kompozisyon tarifi]
             - Giriş Sahnesi Video Promptu: [Türkçe, 9:16 dik format, hızlı tempo, net anlatım, mobil izlemeye uygun kompozisyon]
             - Asıl İçerik Sahnesi Video Promptu: [Türkçe, 9:16 dik format, dinamik geçişler, konuya odaklı görsel anlatım]
             - Kapanış Sahnesi Video Promptu: [Türkçe, 9:16 dik format, güçlü CTA, yüksek dikkat çekicilik, temiz kompozisyon]
           - Her prompt yalnızca ilgili bölümü anlatsın ve dik video üretimine uygun olsun.
        """

    elif platform == "Instagram Reels (Video)":
        return """
        Kanca, görsel ve seslendirme önerileriyle bir video fikri üret.
        Tüm çıktı Türkçe olmalıdır.
        Çıktı formatı:
        1. 🎣 **Kanca (Hook - 0-3sn):** İzleyiciyi ekrana kilitleyecek şok edici giriş cümlesi.
        2. 🎬 **Görsel Kurgu / Sahne Akışı:** Kamera açısı, ekranda ne görüneceği, müzik önerisi. Eğer kurgu birden fazla sahneden oluşuyorsa sahne sahne ayır.
        3. 🗣️ **Seslendirme Metni:** Videoda söylenecek senaryo.
        4. 🚀 **Eylem Çağrısı (CTA):** Kaydetme veya yorum almaya yönelik bitiriş cümlesi.
        5. 🛠️ **Üretim Çantası (Önerilen Araçlar):** Kullanıcıların bu içeriği üretmek için kullanabileceği araçlar:
           - Kurgu: CapCut, Premiere
           - AI Seslendirme: ElevenLabs vb.
           - AI Video: Runway, Veo vb.
        6. 🎨 **Görsel/Video AI İçin Hazır Prompt:** Yalnızca Türkçe video üretim promptları yaz.
           - Asla İngilizce prompt yazma.
           - Eğer senaryo birden fazla ana sahneden oluşuyorsa ASLA tek bir birleşik video promptu verme.
           - Bunun yerine her ana sahne için ayrı ayrı prompt yaz ve şu formatı kullan:
             - Sahne 1 Video Promptu: [Türkçe, detaylı, kamera açısı, kamera hareketi, ışık, kompozisyon ve atmosfer tarifi]
             - Sahne 2 Video Promptu: [Türkçe, detaylı, kamera açısı, kamera hareketi, ışık, kompozisyon ve atmosfer tarifi]
           - Eğer tek sahne varsa yine yalnızca Türkçe tek bir video promptu ver.
           - Her prompt, ilgili sahnenin anlatısına birebir uymalı; genel veya birleştirilmiş anlatım kullanma.
        """

    elif platform == "Instagram Post (Kaydırmalı)":
        return """
        Başlık ve kaydırmalı sayfa içerikleriyle bir carousel fikri üret.
        Tüm çıktı Türkçe olmalıdır.
        Çıktı formatı:
        1. 🖼️ **Görsel Tasarım/Başlık:** Görselin üzerinde yazacak vurucu metin.
        2. 📄 **İçerik Akışı (Slide):** Kaydırmalı post için sayfa sayfa metinler. Her slaydı ayrı numaralandır.
        3. 📝 **Açıklama Metni (Caption):** Postun altına yazılacak detaylı açıklama.
        4. 🛠️ **Üretim Çantası (Önerilen Araçlar):** Kullanıcıların bu içeriği üretmek için kullanabileceği araçlar:
           - Tasarım: Canva
           - AI Görsel: Midjourney, DALL-E vb.
        5. 🎨 **Görsel/Video AI İçin Hazır Prompt:** Yalnızca Türkçe görsel üretim promptları yaz.
           - Asla İngilizce prompt yazma.
           - Kaydırmalı post veya birden fazla görsel gerektiren bir yapı varsa ASLA tek bir birleşik görsel promptu verme.
           - Kaç slayt varsa, her slayt için ayrı ayrı prompt yaz ve şu formatı kullan:
             - Slayt 1 Görsel Promptu: [1. slayta özel Türkçe, detaylı, ışık, kompozisyon, stil ve atmosfer tarifi]
             - Slayt 2 Görsel Promptu: [2. slayta özel Türkçe, detaylı, ışık, kompozisyon, stil ve atmosfer tarifi]
           - Her prompt sadece ilgili slaydın içeriğini anlatsın; tüm slaytları tek karede birleştirmeye çalışma.
           - Tek görsel gerekiyorsa yine yalnızca Türkçe tek bir görsel promptu ver.
        """

    elif platform == "Instagram (Hikaye)":
        return """
        Instagram Hikaye formatı için dikkat çekici, hızlı tüketilen ve 15 saniyelik bölümlerden oluşan bir içerik üret.
        Tüm çıktı Türkçe olmalıdır.
        Hikaye akışı 9:16 dikey format mantığına göre planlanmalıdır.
        Çıktı formatı:
        1. 🎣 **Hikaye 1 (0-15 sn):** Güçlü açılış, ilk mesaj ve dikkat çekici giriş.
        2. 📲 **Hikaye 2 (15-30 sn):** Mesajın gelişimi, ana fayda veya ana problem.
        3. 🚀 **Hikaye 3 (30-45 sn):** Güçlü kapanış, yönlendirme ve aksiyon çağrısı.
        4. 🎯 **Etkileşim Sticker Önerileri:** Her hikaye için uygun etkileşim önerileri ver:
           - Anket
           - Soru Kutusu
           - Link
           - Emoji Slider
        5. 🛠️ **Üretim Çantası (Önerilen Araçlar):** Kullanıcıların bu içeriği üretmek için kullanabileceği araçlar:
           - Tasarım/Kurgu: Canva, CapCut
           - AI Seslendirme: ElevenLabs, CapCut Voice vb.
           - AI Görsel/Video: Midjourney, DALL-E, Runway, Veo vb.
        6. 🎨 **Görsel/Video AI İçin Hazır Prompt:** Yalnızca Türkçe, 9:16 dikey formata uygun görsel ve video üretim promptları yaz.
           - Asla İngilizce prompt yazma.
           - Her hikaye bölümü için ayrı ayrı prompt ver; ASLA tüm hikaye akışını tek bir birleşik promptta toplama.
           - Şu formatı kullan:
             - Hikaye 1 Görsel/Video Promptu: [Türkçe, 9:16 dikey format, güçlü giriş, yüksek dikkat çekicilik, ışık ve kompozisyon tarifi]
             - Hikaye 2 Görsel/Video Promptu: [Türkçe, 9:16 dikey format, ana mesajı taşıyan net ve dinamik sahne tarifi]
             - Hikaye 3 Görsel/Video Promptu: [Türkçe, 9:16 dikey format, güçlü CTA, temiz kompozisyon ve kapanış tarifi]
           - Her prompt yalnızca ilgili hikaye bölümünü anlatsın ve dikey mobil izlemeye uygun olsun.
        """

    elif platform == "YouTube Shorts (Kısa Video)":
        return """
        YouTube Shorts için hızlı tüketilen, yüksek tempolu, maksimum 60 saniyelik bir kısa video içeriği üret.
        Tüm çıktı Türkçe olmalıdır.
        Video dik formatta (9:16) planlanmalıdır.
        İlk 3 saniye çok çarpıcı bir kanca ile başlamalıdır.
        Sahne geçişleri hızlı, dinamik ve kısa dikkat süresine uygun olmalıdır.
        Çıktı formatı:
        1. 🎣 **0-3 Saniye: Kanca:** İzleyiciyi anında videoda tutacak güçlü, merak uyandırıcı açılış.
        2. ⚡ **3-15 Saniye: Giriş:** Konunun hızlı ve net kurulumu.
        3. 🔥 **15-45 Saniye: Asıl İçerik:** En değerli bilgi, gösterim veya anlatımın yüksek tempolu akışı.
        4. 🚀 **45-60 Saniye: Kapanış ve Abone Ol Çağrısı:** İzleyiciyi abone olmaya, yorum yapmaya veya videoyu paylaşmaya yönlendiren kısa kapanış.
        5. 🛠️ **Üretim Çantası (Önerilen Araçlar):** Kullanıcıların bu içeriği üretmek için kullanabileceği araçlar:
           - Kurgu: CapCut, Opus Clip, Premiere Pro
           - AI Seslendirme: ElevenLabs, CapCut Voice, Adobe Podcast vb.
           - AI Video/Görsel: Runway, Veo, Midjourney, DALL-E vb.
        6. 🎨 **Görsel/Video AI İçin Hazır Prompt:** Yalnızca Türkçe, dik formata uygun (9:16) görsel ve video üretim promptları yaz.
           - Asla İngilizce prompt yazma.
           - Her bölüm için ayrı ayrı prompt ver; ASLA tüm Shorts akışını tek bir birleşik promptta toplama.
           - Hızlı geçişleri, mobil ekran uyumunu, yakın planları ve yüksek dikkat çekiciliği promptlara dahil et.
           - Şu formatı kullan:
             - Kanca Sahnesi Video Promptu: [Türkçe, 9:16 dik format, çok çarpıcı açılış, hızlı kamera hareketi, güçlü ışık ve kompozisyon tarifi]
             - Giriş Sahnesi Video Promptu: [Türkçe, 9:16 dik format, hızlı tempo, net anlatım, mobil izlemeye uygun kompozisyon]
             - Asıl İçerik Sahnesi Video Promptu: [Türkçe, 9:16 dik format, dinamik geçişler, konuya odaklı görsel anlatım]
             - Kapanış Sahnesi Video Promptu: [Türkçe, 9:16 dik format, güçlü CTA, yüksek dikkat çekicilik, temiz kompozisyon]
           - Her prompt yalnızca ilgili bölümü anlatsın ve dik video üretimine uygun olsun.
        """

    elif platform == "YouTube (Uzun Video)":
        return """
        YouTube için profesyonel, uzun formatlı bir video içeriği üret.
        Tüm çıktı Türkçe olmalıdır.
        Çıktı formatı:
        1. 🎣 **Çarpıcı Giriş (Hook):** İlk 5-15 saniyede izleyiciyi videoda tutacak güçlü açılış cümlesi.
        2. 👋 **İntro:** Kanalın veya konuşmacının kısa girişi ve videonun vaadi.
        3. ⏱️ **Ana Bölümler (Timestamps ile):** Konuyu mantıklı başlıklara böl, her ana bölüm için zaman damgası ver ve içerik akışını maddeler halinde açıkla.
        4. 🚀 **Kapanış (Call to Action):** Abonelik, yorum veya izlemeye devam etmeye yönlendiren güçlü kapanış.
        5. 🛠️ **Üretim Çantası (Önerilen Araçlar):** Kullanıcıların bu içeriği üretmek için kullanabileceği araçlar:
           - Kurgu: Premiere Pro, DaVinci Resolve, CapCut Desktop
           - AI Ses: ElevenLabs, Adobe Podcast vb.
           - Küçük Resim/Thumbnail: Canva, Photoshop, Midjourney, DALL-E vb.
        6. 🎨 **Görsel/Video AI İçin Hazır Prompt:** Yalnızca Türkçe görsel ve video üretim promptları yaz.
           - Asla İngilizce prompt yazma.
           - Eğer video birden fazla ana sahne veya bölüm içeriyorsa ASLA tek bir birleşik prompt verme.
           - Her ana sahne veya bölüm için ayrı ayrı prompt yaz ve şu formatı kullan:
             - Sahne 1 Video Promptu: [Türkçe, detaylı, kamera açısı, kamera hareketi, ışık, kompozisyon ve atmosfer tarifi]
             - Sahne 2 Video Promptu: [Türkçe, detaylı, kamera açısı, kamera hareketi, ışık, kompozisyon ve atmosfer tarifi]
           - Eğer videoda ayrıca thumbnail önerisi gerekiyorsa ayrı bir satırda şu formatı kullan:
             - Thumbnail Görsel Promptu: [Türkçe, dikkat çekici, yüksek kontrastlı, YouTube küçük resmi için uygun kompozisyon tarifi]
           - Her prompt sadece ilgili sahneyi veya görseli anlatsın; tüm akışı tek promptta birleştirme.
        """

    elif platform == "Twitter/X (Flood)":
        return """
        Zincirleme tweet yapısında kısa ve vurucu bir içerik üret.
        Tüm çıktı Türkçe olmalıdır.
        Çıktı formatı:
        1. 🖼️ **Başlık:** İlk tweet için dikkat çekici bir başlık.
        2. 📄 **Tweet Zinciri:** Zincirleme tweetler halinde içerik akışı.
        3. 📝 **Son Tweet:** Zinciri bitiren güçlü bir çağrı veya özet.
        """

    elif platform == "LinkedIn (Profesyonel)":
        return """
        Kurumsal bir giriş, gelişme ve 'Daha fazlası' vurgusuyla bir içerik üret.
        Tüm çıktı Türkçe olmalıdır.
        Çıktı formatı:
        1. 🖼️ **Başlık:** Gönderinin dikkat çekici başlığı.
        2. 📄 **İçerik Akışı:** Giriş, gelişme ve sonuç bölümleriyle profesyonel bir metin.
        3. 📝 **Açıklama Metni (Caption):** Gönderinin altına yazılacak detaylı açıklama.
        """

    return ""


def get_duration_field_config(platform):
    if platform == "YouTube (Uzun Video)":
        return "Video Süresi", ["3-5 Dakika", "5-10 Dakika", "10+ Dakika"]

    short_video_keywords = ("shorts", "reels", "tiktok")
    if any(keyword in platform.lower() for keyword in short_video_keywords):
        return "Video Süresi", ["15-30 Saniye", "30-60 Saniye"]

    return "İçerik Uzunluğu", ["Kısa (Vurucu)", "Orta", "Uzun (Detaylı Bilgi)"]


def get_algorithm_hacks_prompt(platform):
    platform_lower = platform.lower()

    if platform == "YouTube (Uzun Video)":
        return (
            "Platform algoritma hackleri (YouTube Uzun Video): "
            "İzleyici tutma (retention) taktiklerini uygula, her 2 dakikada bir dikkat yenileyici öge planla, "
            "SEO uyumlu başlık ve thumbnail fikirleri üret."
        )

    if any(keyword in platform_lower for keyword in ("shorts", "reels", "tiktok")):
        return (
            "Platform algoritma hackleri (TikTok/Reels/Shorts): "
            "İlk 3 saniye kuralına uygun güçlü hook üret, hızlı sahne geçişleri tasarla, "
            "videoyu loop hissiyle yeniden izletmeye teşvik edecek kapanış taktikleri kullan."
        )

    if any(keyword in platform_lower for keyword in ("twitter", "x", "linkedin")):
        return (
            "Platform algoritma hackleri (X/LinkedIn): "
            "Dikkat çekici ilk satır yaz, okumayı kolaylaştıran boşluklu paragraf düzeni kur, "
            "yorum/retweet/etkileşimi tetikleyen sorularla bitir."
        )

    return "Seçilen platformun tüketim alışkanlıklarına uygun, yüksek etkileşim odaklı bir kurgu üret."


def extract_json_payload(raw_text):
    cleaned_text = raw_text.strip()

    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?", "", cleaned_text).strip()
        cleaned_text = re.sub(r"```$", "", cleaned_text).strip()

    json_array_match = re.search(r"\[.*\]", cleaned_text, re.DOTALL)
    if json_array_match:
        cleaned_text = json_array_match.group(0)

    return json.loads(cleaned_text)


def fetch_trend_radar_items(model_name):
    fallback_items = [
        {
            "baslik": "Yapay Zeka ile Günlük Verimlilik Challenge",
            "neden_viral": "Hem çalışanlar hem öğrenciler hızla uygulanabilen verimlilik tüyolarına yoğun ilgi gösteriyor.",
            "onerilen_platform": "TikTok & Reels",
            "onerilen_strateji": "Önce-sonra formatı ve hızlı kesimlerle 30 saniyelik mini dönüşüm hikayeleri.",
            "emoji": "🤖",
        },
        {
            "baslik": "Sokak Röportajı: Teknoloji Alışkanlıkları",
            "neden_viral": "Gerçek insan tepkileri ve sürpriz cevaplar yüksek yorum ve paylaşım getiriyor.",
            "onerilen_platform": "YouTube Shorts & Reels",
            "onerilen_strateji": "İlk 2 saniyede iddialı soru, devamında art arda hızlı cevap kolajı.",
            "emoji": "🎤",
        },
        {
            "baslik": "Mikro Girişim Fikirleri 2026",
            "neden_viral": "Ek gelir ve yan iş temaları belirsizlik dönemlerinde güçlü etkileşim alıyor.",
            "onerilen_platform": "LinkedIn & Instagram Carousel",
            "onerilen_strateji": "Sayfa sayfa net formül + son slaytta yapılacaklar listesi.",
            "emoji": "💼",
        },
        {
            "baslik": "1 Dakikada Araç İncelemesi",
            "neden_viral": "Kısa, net ve kıyaslamalı içerikler satın alma kararını hızlandırdığı için çok izleniyor.",
            "onerilen_platform": "TikTok & YouTube Shorts",
            "onerilen_strateji": "Puan tablosu, artı-eksi karşılaştırma ve güçlü kapanış CTA.",
            "emoji": "⚡",
        },
        {
            "baslik": "Haftanın Dijital Gündemi",
            "neden_viral": "Gündem özetleri takipçilere zaman kazandırdığı için düzenli geri dönüş yaratıyor.",
            "onerilen_platform": "Twitter/X & LinkedIn",
            "onerilen_strateji": "3 sıcak gelişme + 1 öngörü formatıyla seri içerik üretimi.",
            "emoji": "🔥",
        },
        {
            "baslik": "Sessiz Vlog: Gerçekçi Çalışma Günü",
            "neden_viral": "Abartısız günlük rutin içerikleri yüksek izlenme süresi ve yorum topluyor.",
            "onerilen_platform": "YouTube Shorts & Reels",
            "onerilen_strateji": "Altyazı odaklı, lo-fi müzikli, 5 sahneli minimal akış.",
            "emoji": "🎧",
        },
        {
            "baslik": "30 Saniyede Kariyer Hack",
            "neden_viral": "Kısa ve uygulanabilir kariyer ipuçları kaydetme oranını artırıyor.",
            "onerilen_platform": "LinkedIn & TikTok",
            "onerilen_strateji": "İlk satırda güçlü vaat, sonda tek aksiyon adımı.",
            "emoji": "🚀",
        },
        {
            "baslik": "Uygulama Karşılaştırma Düellosu",
            "neden_viral": "İki popüler aracın kıyaslandığı içerikler tartışma ve paylaşım üretiyor.",
            "onerilen_platform": "Reels & Shorts",
            "onerilen_strateji": "Split-screen karşılaştırma ve net kazanan açıklaması.",
            "emoji": "🧪",
        },
        {
            "baslik": "Bir Günde Öğren: Mikro Beceri",
            "neden_viral": "Hızlı öğrenme formatı hem keşfete hem kaydetmeye güçlü çalışıyor.",
            "onerilen_platform": "TikTok & Instagram",
            "onerilen_strateji": "Saat saat ilerleme ve final çıktıyı gösteren mini hikaye.",
            "emoji": "📚",
        },
        {
            "baslik": "Tek Cümlelik Finans Gerçekleri",
            "neden_viral": "Sade ve çarpıcı finans içgörüleri yoğun yorum trafiği alıyor.",
            "onerilen_platform": "Twitter/X & LinkedIn",
            "onerilen_strateji": "Her postta tek iddia + kısa veri dayanağı.",
            "emoji": "💸",
        },
        {
            "baslik": "Mit mi Gerçek mi: Teknoloji Efsaneleri",
            "neden_viral": "Yanlış bilinenleri yıkmak izleyicide merak ve etkileşim yaratıyor.",
            "onerilen_platform": "YouTube Shorts & TikTok",
            "onerilen_strateji": "Her videoda tek mit, 3 kanıt ve güçlü kapanış cümlesi.",
            "emoji": "🧠",
        },
        {
            "baslik": "Haftalık İçerik Menü Planı",
            "neden_viral": "Hazır şablon içerikler üreticilere zaman kazandırdığı için düzenli takip sağlıyor.",
            "onerilen_platform": "Instagram Carousel & LinkedIn",
            "onerilen_strateji": "Pazartesi-cuma içerik planı ve bonus CTA şablonu.",
            "emoji": "🗓️",
        },
        {
            "baslik": "Sıfırdan Marka Hikayesi Serisi",
            "neden_viral": "Girişim ve kişisel marka yolculukları güçlü duygusal bağ kurduğu için paylaşım getiriyor.",
            "onerilen_platform": "LinkedIn & Reels",
            "onerilen_strateji": "Her bölümde tek dönüm noktası ve cliffhanger kapanış.",
            "emoji": "🏁",
        },
        {
            "baslik": "Ofis Masası Setup Dönüşümü",
            "neden_viral": "Öncesi-sonrası içerikleri görsel tatmin sunduğu için izlenme süresini artırıyor.",
            "onerilen_platform": "TikTok & Shorts",
            "onerilen_strateji": "5 adımda dönüşüm, her adımda fiyat/etki notu.",
            "emoji": "🖥️",
        },
        {
            "baslik": "Bu Hafta Ne Öğrendim?",
            "neden_viral": "Kısa öğrenim özeti formatı hem bireysel hem profesyonel kitlede sadakat oluşturuyor.",
            "onerilen_platform": "Twitter/X & LinkedIn",
            "onerilen_strateji": "3 madde öğrenim + 1 uygulanabilir aksiyon çağrısı.",
            "emoji": "📝",
        },
    ]

    trend_prompt = """
    Bir sosyal medya trend editörü gibi davran.
    Yalnızca Türkçe yaz.
    ZORUNLU KURAL: Tam olarak 15 farklı trend önermek ZORUNDASIN. 14 veya daha az olamaz! Asla yarıda kesme.
    ÇIKTI FORMATI: Lütfen sadece aşağıdaki gibi net bir numaralandırma ile yanıt ver. Ekstra giriş/çıkış cümlesi kullanma.
    Güvenli parse için ÇIKTIYI SADECE JSON dizi formatında ver (başında/sonunda açıklama metni yazma).
    Dünyada ve Türkiye'de şu an viral olmaya aday tam olarak 15 farklı, taze ve niş trend üret.

    Her öğe şu alanları içermeli:
    - sira_no
    - baslik
    - neden_viral
    - onerilen_platform
    - onerilen_strateji
    - emoji
    """

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(trend_prompt)
        response_text = getattr(response, "text", "") or ""
        parsed_items = extract_json_payload(response_text)

        if not isinstance(parsed_items, list):
            return fallback_items[:15]

        normalized_items = []
        for item in parsed_items[:15]:
            if not isinstance(item, dict):
                continue

            normalized_items.append(
                {
                    "baslik": str(item.get("baslik", "Trend Başlığı")).strip(),
                    "neden_viral": str(item.get("neden_viral", "Bu trend yüksek etkileşim potansiyeli taşıyor.")).strip(),
                    "onerilen_platform": str(item.get("onerilen_platform", "TikTok & Reels")).strip(),
                    "onerilen_strateji": str(item.get("onerilen_strateji", "Kısa, hızlı ve kanca odaklı içerik.")).strip(),
                    "emoji": str(item.get("emoji", "✨")).strip() or "✨",
                }
            )

        if len(normalized_items) < 15:
            needed = 15 - len(normalized_items)
            normalized_items.extend(fallback_items[:needed])

        return normalized_items[:15] if normalized_items else fallback_items[:15]

    except Exception:
        return fallback_items[:15]


def sync_saved_api_key_from_widget():
    st.session_state.saved_api_key = st.session_state.get("api_input_widget", "").strip()


def send_to_studio(topic):
    st.session_state.viral_topic = topic
    st.session_state.page = "✨ İçerik Stüdyosu"


PLATFORM_OPTIONS = [
    "Instagram Reels (Video)",
    "TikTok (Kısa Video)",
    "Instagram Post (Kaydırmalı)",
    "Instagram (Hikaye)",
    "YouTube Shorts (Kısa Video)",
    "YouTube (Uzun Video)",
    "Twitter/X (Flood)",
    "LinkedIn (Profesyonel)",
]
HEDEF_KITLE_OPTIONS = [
    "Geniş Kitleler (Genel İzleyici)",
    "Sektör Profesyonelleri ve Uzmanlar (B2B)",
    "Gençler ve Öğrenciler",
    "Girişimciler ve İşletme Sahipleri",
    "Hobi ve Kişisel Gelişim Meraklıları",
]
ICERIK_TONU_OPTIONS = [
    "Eğitici ve İlham Verici",
    "Samimi ve Sohbet Havasında",
    "Kurumsal ve Bilgi Odaklı",
    "Eğlenceli ve Dinamik",
]
VIRAL_STRATEJI_OPTIONS = [
    "Hikaye Anlatımı (Kişisel/Kurumsal Deneyim)",
    "Veri ve İstatistik Odaklı (Kanıta Dayalı)",
    "Sorun ve Çözüm Sunan (Fayda Odaklı)",
    "Trend ve Gündem Odaklı (Güncel)",
]


def reset_app_state():
    st.session_state.saved_api_key = ""
    st.session_state.api_input_widget = ""
    st.session_state.history = []
    st.session_state.platform = PLATFORM_OPTIONS[0]
    _, default_duration_options = get_duration_field_config(st.session_state.platform)
    st.session_state.sure_uzunluk = default_duration_options[0]
    st.session_state.hedef_kitle = HEDEF_KITLE_OPTIONS[0]
    st.session_state.icerik_tonu = ICERIK_TONU_OPTIONS[0]
    st.session_state.viral_strateji = VIRAL_STRATEJI_OPTIONS[0]
    st.session_state.konu_input = ""
    st.session_state["kalici_platform"] = PLATFORM_OPTIONS[0]
    _, _kf_reset = get_duration_field_config(PLATFORM_OPTIONS[0])
    st.session_state["kalici_format"] = _kf_reset[0]


# Sayfa yapılandırması
st.set_page_config(page_title="Viral Sosyal Medya Stratejisti", layout="wide")
apply_mobile_first_styles()


# Session State ile API Key ve ayarları yönetme
if "saved_api_key" not in st.session_state:
    st.session_state.saved_api_key = st.session_state.get("api_key", "")
if "api_input_widget" not in st.session_state:
    st.session_state.api_input_widget = st.session_state.saved_api_key
if "history" not in st.session_state:
    st.session_state.history = []
if "platform" not in st.session_state:
    st.session_state.platform = PLATFORM_OPTIONS[0]
if "sure_uzunluk" not in st.session_state:
    _, default_duration_options = get_duration_field_config(st.session_state.platform)
    st.session_state.sure_uzunluk = default_duration_options[0]
if "hedef_kitle" not in st.session_state:
    st.session_state.hedef_kitle = HEDEF_KITLE_OPTIONS[0]
if "icerik_tonu" not in st.session_state:
    st.session_state.icerik_tonu = ICERIK_TONU_OPTIONS[0]
if "viral_strateji" not in st.session_state:
    st.session_state.viral_strateji = VIRAL_STRATEJI_OPTIONS[0]
if "konu_input" not in st.session_state:
    st.session_state.konu_input = ""
if "viral_topic" not in st.session_state:
    st.session_state.viral_topic = ""
if "trend_radar_items" not in st.session_state:
    st.session_state.trend_radar_items = []
if "kalici_platform" not in st.session_state:
    st.session_state["kalici_platform"] = PLATFORM_OPTIONS[0]
if "kalici_format" not in st.session_state:
    _, _kf_opts = get_duration_field_config(PLATFORM_OPTIONS[0])
    st.session_state["kalici_format"] = _kf_opts[0]


st.sidebar.title("🚀 AI Tabanlı İçerik Stüdyosu")
page = st.sidebar.radio(
    "Araçlar ve Ayarlar",
    ["✨ İçerik Stüdyosu", "🔥 Trend Radarı", "⚙️ Ayarlar", "📚 Geçmiş"],
    key="page",
)


active_api_key = st.session_state.saved_api_key.strip()
selected_model = None
api_error_message = None
model_warning_message = None

if not active_api_key:
    api_error_message = "Lütfen Ayarlar sayfasından Gemini API anahtarınızı girin."
else:
    try:
        genai.configure(api_key=active_api_key)
    except Exception:
        api_error_message = "API Anahtarı geçersiz veya bağlantı hatası. Lütfen tekrar deneyin."
    else:
        try:
            available_models = genai.list_models()

            for model in available_models:
                if "gemini" in model.name and "generateContent" in model.supported_generation_methods:
                    selected_model = model.name
                    break

            if not selected_model:
                selected_model = "gemini-pro"

        except Exception:
            model_warning_message = "Modeller listelenirken bir hata oluştu. Varsayılan model kullanılacak."
            selected_model = "gemini-pro"

api_ready = bool(active_api_key and not api_error_message and selected_model)
# Kalıcı gölge değişkenlerden oku — widget unmount'tan bağımsız
platform = st.session_state["kalici_platform"]
hedef_kitle = st.session_state.get("hedef_kitle", HEDEF_KITLE_OPTIONS[0])
icerik_tonu = st.session_state.get("icerik_tonu", ICERIK_TONU_OPTIONS[0])
viral_strateji = st.session_state.get("viral_strateji", VIRAL_STRATEJI_OPTIONS[0])
sure_uzunluk = st.session_state["kalici_format"]

# sure_uzunluk'un mevcut platform için geçerli olduğunu doğrula
_, _gecerli_sure_secenekleri = get_duration_field_config(platform)
if sure_uzunluk not in _gecerli_sure_secenekleri:
    sure_uzunluk = _gecerli_sure_secenekleri[0]
    st.session_state["kalici_format"] = sure_uzunluk

if page == "⚙️ Ayarlar":
    st.title("⚙️ Ayarlar")
    st.caption("Platform, strateji ve API yapılandırmasını buradan yönetebilirsiniz.")

    col_left, col_right = st.columns(2)

    with col_left:
        _p_idx = PLATFORM_OPTIONS.index(st.session_state["kalici_platform"]) if st.session_state["kalici_platform"] in PLATFORM_OPTIONS else 0
        _secilen_platform = st.selectbox("🌐 Platform", PLATFORM_OPTIONS, index=_p_idx)
        st.session_state["kalici_platform"] = _secilen_platform

        duration_label, duration_options = get_duration_field_config(_secilen_platform)
        _f_idx = duration_options.index(st.session_state["kalici_format"]) if st.session_state["kalici_format"] in duration_options else 0
        _secilen_format = st.selectbox(duration_label, duration_options, index=_f_idx)
        st.session_state["kalici_format"] = _secilen_format

        st.selectbox("🎯 Hedef Kitle", HEDEF_KITLE_OPTIONS, key="hedef_kitle")

    with col_right:
        st.selectbox("🎭 İçerik Tonu", ICERIK_TONU_OPTIONS, key="icerik_tonu")
        st.selectbox("🌟 Viral Strateji", VIRAL_STRATEJI_OPTIONS, key="viral_strateji")

    st.divider()
    st.subheader("🔑 Kendi Gemini API Anahtarınız (İsteğe Bağlı)")
    st.caption("Kotalara takılmadan sınırsız kullanım için kendi anahtarınızı girebilirsiniz.")
    st.text_input(
        "Gemini API Anahtarı",
        type="password",
        key="api_input_widget",
        on_change=sync_saved_api_key_from_widget,
    )

    if api_ready:
        if st.session_state.saved_api_key.strip():
            st.success("Kendi API anahtarınız aktif.")
        else:
            st.info("Sistem varsayılan API anahtarı aktif.")
    elif active_api_key:
        st.error(api_error_message)
    else:
        st.warning(api_error_message)

    if model_warning_message:
        st.warning(model_warning_message)

    st.divider()
    if st.button("Ayarları Sıfırla", use_container_width=True):
        reset_app_state()
        st.rerun()

elif page == "✨ İçerik Stüdyosu":
    st.title("✨ İçerik Stüdyosu")
    st.caption("Fikrinizi yazın, profesyonel içerik akışını tek tıkla üretin.")

    st.info(
        f"📌 **Aktif Ayarlar** → "
        f"🌐 {platform} · "
        f"⏱️ {sure_uzunluk} · "
        f"🎯 {hedef_kitle} · "
        f"🎭 {icerik_tonu}\n\n"
        "_Değiştirmek için kenar menüden ⚙️ **Ayarlar** sekmesine gidin._"
    )

    if st.session_state.viral_topic:
        st.session_state.konu_input = st.session_state.viral_topic
        st.session_state.viral_topic = ""

    konu = st.text_area(
        "Ne hakkında içerik üretmek istiyorsun?",
        placeholder="Örneğin: Yazılım ekiplerinde verimlilik artırma, girişimcilikte ilk müşteri bulma, kişisel gelişimde odak yönetimi",
        height=200,
        key="konu_input",
    )

    if st.button("🚀 Üret", type="primary", use_container_width=True):
        if not api_ready:
            st.warning(api_error_message or "Üretim için önce Ayarlar sayfasından geçerli bir Gemini API anahtarı sağlayın.")
            st.stop()

        if not konu.strip():
            st.warning("Lütfen bir konu girin.")
            st.stop()

        # Widget unmount koruması: rerun sırasında kalıcı hafızadan doğrudan oku
        platform = st.session_state["kalici_platform"]
        sure_uzunluk = st.session_state["kalici_format"]
        platform_lower = platform.lower()

        if any(k in platform_lower for k in ("tiktok", "reels", "shorts")):
            platform_kurali = (
                "KISA VİDEO (TikTok/Reels/Shorts) KURALI — KESİNLİKLE UYGULA:\n"
                "- İlk 3 saniyede izleyiciyi kilitleyen bir Hook/Kanca yaz.\n"
                "- Saniye saniye akış ver: (0-3 sn: ...) → (3-8 sn: ...) → (8-30 sn: ...) vb.\n"
                "- Ekran metni (on-screen text) ile seslendirme (voiceover) metnini AYRI AYRI belirt.\n"
                "- Loop hissi yaratacak, enerjisi yüksek bir kapanış tasarla.\n"
                "- Uzun paragraf ve akademik anlatım KESİNLİKLE YASAKTIR."
            )
            cikis_sablonu = """
══════════════════════════════════════════
ZORUNLU ÇIKTI ŞABLONU — SADECE BU FORMATI KULLAN
══════════════════════════════════════════
🔥 [0-3 Sn — Kanca (Hook)]: (İzleyiciyi anında kilitleyen açılış — buraya yaz)

⚡ [3-8 Sn — Giriş / Kurulum]: (Konunun hızlı ve net kurulumu — buraya yaz)

🎬 [8-{sure} Sn — Görsel / Video Akışı]:
  - Sahne 1 (X-X sn): (Ne görüyoruz, kamera açısı, hareket — buraya yaz)
  - Sahne 2 (X-X sn): (Ne görüyoruz, kamera açısı, hareket — buraya yaz)
  - Sahne 3 (X-X sn): (Ne görüyoruz, kamera açısı, hareket — buraya yaz)

🗣️ [Seslendirme / Voiceover Metni]: (Söylenecek tam metin — buraya yaz)

📱 [On-Screen Text / Ekran Yazısı]: (Ekranda flash gösterilecek yazılar — buraya yaz)

🚀 [Kapanış + CTA]: (Loop hissi yaratacak kapanış ve harekete geçirici mesaj — buraya yaz)

🛠️ [Üretim Araçları]: (CapCut, ElevenLabs, Runway vb. öneriler — buraya yaz)

🎨 [AI Görsel/Video Promptu]: (Her sahne için ayrı Türkçe prompt — buraya yaz)
══════════════════════════════════════════"""

        elif any(k in platform_lower for k in ("hikaye", "story")):
            platform_kurali = (
                "INSTAGRAM HİKAYE (Story) KURALI — KESİNLİKLE UYGULA:\n"
                "- ASLA uzun metin yazma! Maksimum 1-3 slaytlık kompakt akış ver.\n"
                "- Her slayt için AYRI AYRI belirt:\n"
                "  1. Görsel fikri (ne gösterilecek)\n"
                "  2. Etkileşim çıkartması: Anket / Soru Kutusu / Link / Emoji Slider\n"
                "  3. Slayt üzerine yazılacak maksimum 2 cümle\n"
                "- Detaylı paragraf ve uzun metin KESİNLİKLE YASAKTIR."
            )
            cikis_sablonu = """
══════════════════════════════════════════
ZORUNLU ÇIKTI ŞABLONU — SADECE BU FORMATI KULLAN (UZUN METİN YASAK)
══════════════════════════════════════════
📱 Slayt 1:
  🖼️ Görsel Fikri   : (Ne gösterilecek — buraya yaz)
  ✍️ Ekran Metni    : (Max 2 cümle — buraya yaz)
  🎯 Etkileşim      : (Anket / Soru Kutusu / Emoji Slider — hangisi uygunsa yaz)

📱 Slayt 2:
  🖼️ Görsel Fikri   : (Ne gösterilecek — buraya yaz)
  ✍️ Ekran Metni    : (Max 2 cümle — buraya yaz)
  🎯 Etkileşim      : (Anket / Soru Kutusu / Link — hangisi uygunsa yaz)

📱 Slayt 3:
  🖼️ Görsel Fikri   : (Ne gösterilecek — buraya yaz)
  ✍️ Ekran Metni    : (Max 2 cümle — buraya yaz)
  🎯 Etkileşim      : (Link / CTA Butonu — buraya yaz)

🎨 [AI Görsel Promptları]: (Her slayt için ayrı Türkçe, 9:16 dikey format — buraya yaz)
══════════════════════════════════════════"""

        elif "youtube" in platform_lower and "uzun" in platform_lower:
            platform_kurali = (
                f"YOUTUBE UZUN VİDEO KURALI — KESİNLİKLE UYGULA (Süre: {sure_uzunluk}):\n"
                "- Dakika/saniye bazlı profesyonel senaryo iskeleti çıkar.\n"
                "- Yapı: Hook (Giriş) → Timestamp'li Ana Bölümler → CTA Kapanış.\n"
                f"- Süre tercihi '{sure_uzunluk}' olduğu için bölüm sayısını ve detay düzeyini buna göre ayarla.\n"
                "- Her bölüm için zaman damgası (00:00, 02:30 vb.) ekle."
            )
            cikis_sablonu = """
══════════════════════════════════════════
ZORUNLU ÇIKTI ŞABLONU — SADECE BU FORMATI KULLAN
══════════════════════════════════════════
🎣 [Giriş / Hook — 00:00-00:15]: (İzleyiciyi tutacak çarpıcı açılış — buraya yaz)

👋 [İntro — 00:15-01:00]: (Kanal/konuşmacı tanıtımı ve videonun vaadi — buraya yaz)

⏱️ [Bölüm 1 — 01:00-0X:XX]: (Başlık ve içerik özeti — buraya yaz)

⏱️ [Bölüm 2 — 0X:XX-0X:XX]: (Başlık ve içerik özeti — buraya yaz)

⏱️ [Bölüm 3 — 0X:XX-0X:XX]: (Başlık ve içerik özeti — buraya yaz)

🚀 [Kapanış + CTA — Son 1 dk]: (Abone ol / beğen / yorum yap çağrısı — buraya yaz)

🛠️ [Üretim Araçları]: (Premiere, DaVinci, ElevenLabs vb. öneriler — buraya yaz)

🎨 [Thumbnail + AI Video Promptları]: (Sahne bazlı Türkçe promptlar — buraya yaz)
══════════════════════════════════════════"""

        elif "linkedin" in platform_lower:
            platform_kurali = (
                "LİNKEDIN KURALI — KESİNLİKLE UYGULA:\n"
                "- Storytelling (hikaye anlatıcılığı) tekniğini kullan.\n"
                "- Her paragraf arasına MUTLAKA boş satır bırak (okunabilirlik için).\n"
                "- Profesyonel ama samimi bir dil; ilk cümle merak uyandırıcı olsun.\n"
                "- Metnin sonuna 3-5 adet sektörel hashtag ekle.\n"
                "- Kuru, listeli veya madde madde format KESİNLİKLE YASAKTIR."
            )
            cikis_sablonu = """
══════════════════════════════════════════
ZORUNLU ÇIKTI ŞABLONU — SADECE BU FORMATI KULLAN
══════════════════════════════════════════
[Çarpıcı Açılış Cümlesi — merak uyandırıcı, tek satır]

[Hikaye / Problem Paragrafı — 2-4 cümle, samimi anlatım]

[Gelişme / Değer Katan İçerik Paragrafı — 2-4 cümle]

[Sonuç / İçgörü / Ders Paragrafı — 2-3 cümle]

[Soru veya Harekete Geçirici Mesaj (CTA) — tek satır]

#Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5
══════════════════════════════════════════"""

        elif any(k in platform_lower for k in ("twitter", "/x", "flood")):
            platform_kurali = (
                "TWITTER/X THREAD KURALI — KESİNLİKLE UYGULA:\n"
                "- Her tweet için 280 karakter sınırına KESİNLİKLE uy.\n"
                "- Thread/Flood formatında ver; her tweeti (1/x), (2/x) şeklinde numaralandır.\n"
                "- En çarpıcı, dikkat çekici cümleyi birinci tweete koy.\n"
                "- Son tweete güçlü bir CTA (eylem çağrısı) veya özet ekle.\n"
                "- 280 karakteri aşan tek tweet üretmek KESİNLİKLE YASAKTIR."
            )
            cikis_sablonu = """
══════════════════════════════════════════
ZORUNLU ÇIKTI ŞABLONU — SADECE BU FORMATI KULLAN (her tweet max 280 karakter)
══════════════════════════════════════════
Tweet 1/X: [En vurucu, dikkat çekici açılış cümlesi — max 280 karakter]

Tweet 2/X: [Gelişme / ilk bilgi — max 280 karakter]

Tweet 3/X: [Gelişme / ikinci bilgi — max 280 karakter]

Tweet 4/X: [Gelişme / üçüncü bilgi — max 280 karakter]

Tweet X/X: [Kapanış + güçlü CTA (beğen, retweet, yorum yap) — max 280 karakter]
══════════════════════════════════════════"""

        elif any(k in platform_lower for k in ("kaydırmalı", "carousel", "post")):
            platform_kurali = (
                "INSTAGRAM CAROUSEL/POST KURALI — KESİNLİKLE UYGULA:\n"
                "- Slayt slayt içerik ver: Slayt 1:, Slayt 2:, ... şeklinde numaralandır.\n"
                "- Her slayt için görsel fikri ve üzerindeki metin ayrı ayrı belirtilmeli.\n"
                "- Caption (açıklama metni) CTA içerecek şekilde ayrıca yaz.\n"
                "- Metnin sonuna 5-10 ilgili hashtag ekle."
            )
            cikis_sablonu = """
══════════════════════════════════════════
ZORUNLU ÇIKTI ŞABLONU — SADECE BU FORMATI KULLAN
══════════════════════════════════════════
🖼️ [Kapak Slaytı Başlığı]: (Görselin üzerinde yazacak vurucu metin — buraya yaz)

📄 Slayt 1: [Görsel Fikri] — [Üzerindeki Metin]
📄 Slayt 2: [Görsel Fikri] — [Üzerindeki Metin]
📄 Slayt 3: [Görsel Fikri] — [Üzerindeki Metin]
📄 Slayt 4: [Görsel Fikri] — [Üzerindeki Metin]
📄 Son Slayt: [CTA Görseli] — [Harekete Geçirici Mesaj]

📝 [Caption / Açıklama Metni]: (Okunabilir, CTA içeren tam metin — buraya yaz)

#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5

🎨 [AI Görsel Promptları]: (Her slayt için ayrı Türkçe prompt — buraya yaz)
══════════════════════════════════════════"""

        else:
            platform_kurali = (
                f"'{platform}' PLATFORMA ÖZEL KURAL — KESİNLİKLE UYGULA:\n"
                "- Seçilen platformun tüketim alışkanlıklarına ve algoritma dinamiklerine uygun içerik üret.\n"
                "- Yüksek etkileşim ve kaydetme/paylaşma oranı hedefle."
            )
            cikis_sablonu = ""

        prompt = f"""
⚠️⚠️ ZORUNLU PLATFORM TALİMATLARI — BU KURALLARA %100 İTAAT ETMELİSİN ⚠️⚠️

KULLANICININ SEÇTİĞİ PLATFORM : {platform}
KULLANICININ SEÇTİĞİ SÜRE/FORMAT: {sure_uzunluk}

{platform_kurali}

UYARI: Yukarıdaki kurallara aykırı hiçbir format veya yapı KULLANAMAZSIN.
Seçilen platform için geçerli olmayan çıktı biçimleri KESİNLİKLE YASAKTIR.

══════════════════════════════════════════
İÇERİK BİLGİLERİ
══════════════════════════════════════════
Konu         : {konu}
Hedef Kitle  : {hedef_kitle}
İçerik Tonu  : {icerik_tonu}
Viral Strateji: {viral_strateji}

Tüm çıktıları Türkçe üret.
Görsel ve video AI promptlarında asla İngilizce kullanma.
Sen üst düzey bir Sosyal Medya ve Algoritma Uzmanısın.

{get_algorithm_hacks_prompt(platform)}

{get_platform_specific_prompt(platform)}

{cikis_sablonu}
        """

        try:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(selected_model)

            st.divider()
            st.subheader("Üretilen İçerik Fikri")
            main_output_placeholder = st.empty()

            with st.status("🤖 İçerik Mühendisliği Başladı...", expanded=True) as status:
                status.write("🔍 Seçilen platform algoritması analiz ediliyor...")
                time.sleep(1)
                status.write("🎯 Viral kancalar (Hook) ve süre dinamikleri hesaplanıyor...")
                time.sleep(1)
                status.write("✍️ Yapay zeka senaryoyu stüdyoya döküyor...")
                time.sleep(1)

                output = generate_content_with_retry(
                    model,
                    prompt,
                    safety_settings,
                    main_output_placeholder,
                    show_spinner=False,
                )

                status.update(label="✅ İçerik Başarıyla Üretildi!", state="complete", expanded=False)

            main_content, _ = split_output_sections(output)
            if main_content:
                main_output_placeholder.markdown(main_content)
            else:
                main_output_placeholder.empty()

            if output:
                st.toast("✅ İçeriğiniz başarıyla oluşturuldu!", icon="🎉")
                render_output_details(
                    output,
                    download_key=f"current_output_download_{len(st.session_state.history) + 1}",
                    show_main_content=False,
                )
                st.session_state.history.append(
                    {
                        "id": len(st.session_state.history) + 1,
                        "konu": konu,
                        "platform": platform,
                        "icerik": output,
                    }
                )
            else:
                st.warning("⚠️ İçerik filtreye takıldı, lütfen konuyu veya tonu değiştirip tekrar deneyin.")

        except Exception as error:
            if is_rate_limit_error(error):
                st.warning(
                    "Şu anda çok fazla kişi içerik üretiyor ve sunucularımız limitlerine ulaştı. Lütfen 1 dakika sonra tekrar deneyin veya Ayarlar sayfasından kendi API anahtarınızı girerek sınırsızca kullanın."
                )
            else:
                st.error(f"Bir hata oluştu: {str(error)}")

elif page == "🔥 Trend Radarı":
    st.title("⚡ Sosyal Medya Magazini & Trend Analizi")
    st.caption("Güncel trendleri tek bakışta inceleyin ve tek tıkla stüdyoya taşıyın.")

    if not api_ready:
        st.warning(api_error_message or "Trend analizi için geçerli bir API anahtarı gerekiyor.")
    else:
        if not st.session_state.trend_radar_items:
            with st.status("🛰️ Trend sinyalleri toplanıyor...", expanded=True) as trend_status:
                trend_status.write("🌍 Türkiye ve global gündem taranıyor...")
                time.sleep(1)
                trend_status.write("📈 Viral potansiyel skorları hazırlanıyor...")
                time.sleep(1)
                trend_status.write("🧠 Editöryal trend kartları oluşturuluyor...")
                time.sleep(1)

                st.session_state.trend_radar_items = fetch_trend_radar_items(selected_model)
                trend_status.update(label="✅ Trend raporu hazır!", state="complete", expanded=False)

        if st.button("🔄 Trendleri Yenile", use_container_width=True):
            with st.status("♻️ Trend listesi güncelleniyor...", expanded=False) as refresh_status:
                st.session_state.trend_radar_items = fetch_trend_radar_items(selected_model)
                refresh_status.update(label="✅ Trendler güncellendi", state="complete", expanded=False)

        for index, trend in enumerate(st.session_state.trend_radar_items, start=1):
            emoji = trend.get("emoji", "✨")
            trend_title = trend.get("baslik", "Trend Başlığı")
            trend_why = trend.get("neden_viral", "Bu trend yüksek etkileşim potansiyeli taşıyor.")
            trend_platform = trend.get("onerilen_platform", "TikTok & Reels")
            trend_strategy = trend.get("onerilen_strateji", "Kısa, hızlı ve kanca odaklı içerik.")

            with st.expander(f"{emoji} {trend_title}"):
                st.markdown(f"**Neden Viral?** {trend_why}")
                st.markdown(f"**Önerilen Platform:** {trend_platform}")
                st.markdown(f"**Önerilen Strateji:** {trend_strategy}")

                if st.button(
                    "🚀 Bu Fikri Stüdyoya Gönder",
                    key=f"send_trend_to_studio_{index}",
                    use_container_width=True,
                    on_click=send_to_studio,
                    args=(trend_title,),
                ):
                    pass

elif page == "📚 Geçmiş":
    st.title("📚 Geçmiş")
    st.caption("Daha önce üretilen içerikleri tekrar görüntüleyin ve indirin.")

    total_generation_count = len(st.session_state.history)
    last_platform = st.session_state.history[-1]["platform"] if st.session_state.history else "Yok"
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Toplam Üretim", total_generation_count)
    metric_col_2.metric("Son Kullanılan Platform", last_platform)
    metric_col_3.metric("Sistem Durumu", "Aktif 🟢")

    st.divider()

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            expander_title = f"📝 {item['konu']} · {item['platform']}"
            with st.expander(expander_title):
                render_output_details(
                    item["icerik"],
                    download_key=f"history_download_{item['id']}",
                )
    else:
        st.info("Henüz bir içerik üretilmedi.")
