use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Reason {
    UrlSpam = 0,
    PromoPhrase = 1,
    PromoHandle = 2,
    PromoContact = 3,
    CtaSentence = 4,
    Boilerplate = 5,
    CookieBanner = 6,
    NavMenu = 7,
    SymbolHeavy = 8,
    Repetition = 9,
    TooShort = 10,
    LowAlpha = 11,
    GpuGarbage = 12,
    WrongLang = 13,
    DuplicateExact = 14,
    DuplicateNear = 15,
    PiiDropped = 16,
    EmptyAfterClean = 17,
    LowScore = 18,
}

impl Reason {
    pub const ALL: [Reason; 19] = [
        Reason::UrlSpam,
        Reason::PromoPhrase,
        Reason::PromoHandle,
        Reason::PromoContact,
        Reason::CtaSentence,
        Reason::Boilerplate,
        Reason::CookieBanner,
        Reason::NavMenu,
        Reason::SymbolHeavy,
        Reason::Repetition,
        Reason::TooShort,
        Reason::LowAlpha,
        Reason::GpuGarbage,
        Reason::WrongLang,
        Reason::DuplicateExact,
        Reason::DuplicateNear,
        Reason::PiiDropped,
        Reason::EmptyAfterClean,
        Reason::LowScore,
    ];

    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Reason::UrlSpam => "url_spam",
            Reason::PromoPhrase => "promo_phrase",
            Reason::PromoHandle => "promo_handle",
            Reason::PromoContact => "promo_contact",
            Reason::CtaSentence => "cta_sentence",
            Reason::Boilerplate => "boilerplate",
            Reason::CookieBanner => "cookie_banner",
            Reason::NavMenu => "nav_menu",
            Reason::SymbolHeavy => "symbol_heavy",
            Reason::Repetition => "repetition",
            Reason::TooShort => "too_short",
            Reason::LowAlpha => "low_alpha",
            Reason::GpuGarbage => "gpu_garbage",
            Reason::WrongLang => "wrong_lang",
            Reason::DuplicateExact => "duplicate_exact",
            Reason::DuplicateNear => "duplicate_near",
            Reason::PiiDropped => "pii_dropped",
            Reason::EmptyAfterClean => "empty_after_clean",
            Reason::LowScore => "low_score",
        }
    }

    #[inline]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "url_spam" => Some(Reason::UrlSpam),
            "promo_phrase" => Some(Reason::PromoPhrase),
            "promo_handle" => Some(Reason::PromoHandle),
            "promo_contact" => Some(Reason::PromoContact),
            "cta_sentence" => Some(Reason::CtaSentence),
            "boilerplate" => Some(Reason::Boilerplate),
            "cookie_banner" => Some(Reason::CookieBanner),
            "nav_menu" => Some(Reason::NavMenu),
            "symbol_heavy" => Some(Reason::SymbolHeavy),
            "repetition" => Some(Reason::Repetition),
            "too_short" => Some(Reason::TooShort),
            "low_alpha" => Some(Reason::LowAlpha),
            "gpu_garbage" => Some(Reason::GpuGarbage),
            "wrong_lang" => Some(Reason::WrongLang),
            "duplicate_exact" => Some(Reason::DuplicateExact),
            "duplicate_near" => Some(Reason::DuplicateNear),
            "pii_dropped" => Some(Reason::PiiDropped),
            "empty_after_clean" => Some(Reason::EmptyAfterClean),
            "low_score" => Some(Reason::LowScore),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ReasonSet(pub u32);

impl ReasonSet {
    #[inline]
    pub fn new() -> Self {
        Self(0)
    }

    #[inline]
    pub fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub fn insert(&mut self, reason: Reason) {
        self.0 |= 1u32 << (reason as u8);
    }

    #[inline]
    pub fn remove(&mut self, reason: Reason) {
        self.0 &= !(1u32 << (reason as u8));
    }

    #[inline]
    pub fn contains(&self, reason: Reason) -> bool {
        (self.0 & (1u32 << (reason as u8))) != 0
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn iter(&self) -> ReasonSetIter {
        ReasonSetIter {
            bits: self.0,
            index: 0,
        }
    }

    pub fn to_vec(&self) -> Vec<String> {
        self.iter().map(|r| r.as_str().to_string()).collect()
    }

    pub fn to_reasons(&self) -> Vec<Reason> {
        self.iter().collect()
    }

    pub fn union(&self, other: ReasonSet) -> ReasonSet {
        ReasonSet(self.0 | other.0)
    }
}

impl std::ops::BitOr for ReasonSet {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ReasonSet {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

pub struct ReasonSetIter {
    bits: u32,
    index: u8,
}

impl Iterator for ReasonSetIter {
    type Item = Reason;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < 19 {
            let mask = 1u32 << self.index;
            let current = self.index;
            self.index += 1;
            if (self.bits & mask) != 0 {
                return match current {
                    0 => Some(Reason::UrlSpam),
                    1 => Some(Reason::PromoPhrase),
                    2 => Some(Reason::PromoHandle),
                    3 => Some(Reason::PromoContact),
                    4 => Some(Reason::CtaSentence),
                    5 => Some(Reason::Boilerplate),
                    6 => Some(Reason::CookieBanner),
                    7 => Some(Reason::NavMenu),
                    8 => Some(Reason::SymbolHeavy),
                    9 => Some(Reason::Repetition),
                    10 => Some(Reason::TooShort),
                    11 => Some(Reason::LowAlpha),
                    12 => Some(Reason::GpuGarbage),
                    13 => Some(Reason::WrongLang),
                    14 => Some(Reason::DuplicateExact),
                    15 => Some(Reason::DuplicateNear),
                    16 => Some(Reason::PiiDropped),
                    17 => Some(Reason::EmptyAfterClean),
                    18 => Some(Reason::LowScore),
                    _ => None,
                };
            }
        }
        None
    }
}
