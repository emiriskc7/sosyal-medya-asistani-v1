import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold  # Güvenlik ayarları için gerekli import

# Sayfa yapılandırması
st.set_page_config(page_title="Viral Sosyal Medya Stratejisti", layout="wide")

# Session State ile API Key ve ayarları yönetme
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "settings_reset" not in st.session_state:
    st.session_state.settings_reset = False
if "history" not in st.session_state:
    st.session_state.history = []  # Oturum geçmişi için liste

# Sol menü (Sidebar)
with st.sidebar:
    st.title("🔑 API Ayarları")
    st.session_state.api_key = st.text_input("API Anahtarı", type="password", value=st.session_state.api_key)
    if st.button("Ayarları Sıfırla"):
        st.session_state.settings_reset = True
        st.session_state.api_key = ""
        st.session_state.history = []  # Geçmişi sıfırla
        st.rerun()

    # Geçmiş Fikirler
    st.markdown("### 📝 Geçmiş Fikirler")
    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"{item['konu']} - {item['platform']}"):
                st.markdown(item["icerik"])
    else:
        st.write("Henüz bir fikir üretilmedi.")

# API Key doğrulama
if not st.session_state.api_key:
    st.warning("Lütfen API Anahtarınızı girin.")
    st.stop()

# Google Generative AI yapılandırması
try:
    genai.configure(api_key=st.session_state.api_key)
except Exception as e:
    st.error("API Anahtarı geçersiz veya bağlantı hatası. Lütfen tekrar deneyin.")
    st.stop()

# Akıllı Model Seçimi
try:
    available_models = genai.list_models()
    selected_model = None

    # "gemini" içeren ve "generateContent" özelliğini destekleyen ilk modeli seç
    for model in available_models:
        if "gemini" in model.name and "generateContent" in model.supported_generation_methods:
            selected_model = model.name
            break

    # Eğer uygun bir model bulunamazsa varsayılan olarak "gemini-pro" kullan
    if not selected_model:
        selected_model = "gemini-pro"

except Exception as e:
    st.error("Modeller listelenirken bir hata oluştu. Varsayılan model kullanılacak.")
    selected_model = "gemini-pro"

# Ana ekran (Main)
st.title("📈 Viral Sosyal Medya Stratejisti")
st.write("Tüm sosyal medya platformlarının algoritmasını manipüle edebilecek içerik fikirleri üretin!")

# Kullanıcı seçimleri
col1, col2 = st.columns(2)
with col1:
    konu = st.text_input("📌 Konu", placeholder="Örneğin: Yazılım, Girişimcilik, Kişisel Gelişim")
    platform = st.selectbox(
        "🌐 Platform",
        ["Instagram Reels (Video)", "Instagram Post (Kaydırmalı)", "Twitter/X (Flood)", "LinkedIn (Profesyonel)"]
    )
with col2:
    hedef_kitle = st.selectbox("🎯 Hedef Kitle", ["Yeni Başlayanlar", "Orta Seviye", "Uzmanlar", "Girişimciler"])
    icerik_tonu = st.selectbox("🎭 İçerik Tonu", ["Eğlenceli/Mizahi", "Sert/Eleştirel", "Öğretici/Akademik", "Motive Edici"])

viral_strateji = st.radio(
    "🌟 Viral Strateji",
    [
        "Tartışma Yarat (Yorum Kasma)",
        "Değer Odaklı (Kaydetme Kasma)",
        "Hata & Korku (İzlenme Süresi)",
        "Bizden Biri (Paylaşım/Relatable)"
    ]
)

if st.button("🚀 Fikir Üret"):
    if not konu:
        st.warning("Lütfen bir konu girin.")
        st.stop()

    # Prompt oluşturma
    if platform == "Instagram Reels (Video)":
        platform_specific_prompt = """
        Kanca, görsel ve seslendirme önerileriyle bir video fikri üret.
        Çıktı formatı:
        1. 🎣 **Kanca (Hook - 0-3sn):** İzleyiciyi ekrana kilitleyecek şok edici giriş cümlesi.
        2. 🎬 **Görsel Kurgu:** Kamera açısı, ekranda ne görüneceği, müzik önerisi.
        3. 🗣️ **Seslendirme Metni:** Videoda söylenecek senaryo.
        4. 🚀 **Eylem Çağrısı (CTA):** Kaydetme veya yorum almaya yönelik bitiriş cümlesi.
        """
    elif platform == "Instagram Post (Kaydırmalı)":
        platform_specific_prompt = """
        Başlık ve kaydırmalı sayfa içerikleriyle bir carousel fikri üret.
        Çıktı formatı:
        1. 🖼️ **Görsel Tasarım/Başlık:** Görselin üzerinde yazacak vurucu metin.
        2. 📄 **İçerik Akışı (Slide):** Kaydırmalı post için sayfa sayfa metinler.
        3. 📝 **Açıklama Metni (Caption):** Postun altına yazılacak detaylı açıklama.
        """
    elif platform == "Twitter/X (Flood)":
        platform_specific_prompt = """
        Zincirleme tweet yapısında kısa ve vurucu bir içerik üret.
        Çıktı formatı:
        1. 🖼️ **Başlık:** İlk tweet için dikkat çekici bir başlık.
        2. 📄 **Tweet Zinciri:** Zincirleme tweetler halinde içerik akışı.
        3. 📝 **Son Tweet:** Zinciri bitiren güçlü bir çağrı veya özet.
        """
    elif platform == "LinkedIn (Profesyonel)":
        platform_specific_prompt = """
        Kurumsal bir giriş, gelişme ve 'Daha fazlası' vurgusuyla bir içerik üret.
        Çıktı formatı:
        1. 🖼️ **Başlık:** Gönderinin dikkat çekici başlığı.
        2. 📄 **İçerik Akışı:** Giriş, gelişme ve sonuç bölümleriyle profesyonel bir metin.
        3. 📝 **Açıklama Metni (Caption):** Gönderinin altına yazılacak detaylı açıklama.
        """

    prompt = f"""
    Sen bir sosyal medya içerik stratejisti ve uzman bir metin yazarı olarak çalışıyorsun. 
    Kullanıcı sana şu bilgileri verdi:
    - Konu: {konu}
    - Platform: {platform}
    - Hedef Kitle: {hedef_kitle}
    - İçerik Tonu: {icerik_tonu}
    - Viral Strateji: {viral_strateji}

    {platform_specific_prompt}
    """

    try:
        # Güvenlik ayarlarını tanımlama
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Model oluşturma ve içerik üretme
        model = genai.GenerativeModel(selected_model)
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # Yanıt kontrolü
        if response.text:
            output = response.text  # Yanıt okuma kısmı
            st.markdown("### 📊 Üretilen İçerik Fikri")
            st.markdown(output)

            # İndirme butonu
            st.download_button("📥 Bu Fikri İndir", output, file_name="icerik_fikri.txt")

            # Geçmişe kaydetme
            st.session_state.history.append({"konu": konu, "platform": platform, "icerik": output})
        else:
            st.warning("⚠️ İçerik filtreye takıldı, lütfen konuyu veya tonu değiştirip tekrar deneyin.")

    except Exception as e:
        error_message = str(e)
        if "429" in error_message or "Quota" in error_message:
            st.error("API kota sınırına ulaşıldı. Lütfen daha sonra tekrar deneyin.")
        else:
            st.error(f"Bir hata oluştu: {error_message}")
