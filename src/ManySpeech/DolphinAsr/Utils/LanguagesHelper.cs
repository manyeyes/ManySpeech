namespace ManySpeech.DolphinAsr.Utils
{
    internal class LanguagesHelper
    {
        /// <summary>
        /// 基础语言代码映射（如 zh -> 中文）
        /// Key: 语言简码（zh/ja/th 等）
        /// Value: 数组，[0] = 英文名称, [1] = 中文名称
        /// </summary>
        public Dictionary<string, string[]> LanguageCodes { get; set; }

        /// <summary>
        /// 语言+区域代码映射（如 zh-CN -> 中文(普通话)）
        /// Key: 语言-区域简码（zh-CN/ja-JP 等）
        /// Value: 数组，[0] = 英文名称, [1] = 中文名称
        /// </summary>
        public Dictionary<string, string[]> LanguageRegionCodes { get; set; }

        /// <summary>
        /// 初始化语言配置（加载所有预设的语言映射数据）
        /// </summary>
        /// <returns>初始化完成的 LanguageConfig 实例</returns>
        public static LanguagesHelper Initialize()
        {
            return new LanguagesHelper
            {
                // 初始化基础语言代码字典
                LanguageCodes = new Dictionary<string, string[]>
            {
                {"zh", new[] {"Mandarin Chinese", "中文"}},
                {"ja", new[] {"Japanese", "日语"}},
                {"th", new[] {"Thai", "泰语"}},
                {"ru", new[] {"Russian", "俄语"}},
                {"ko", new[] {"Korean", "韩语"}},
                {"id", new[] {"Indonesian", "印度尼西亚语"}},
                {"vi", new[] {"Vietnamese", "越南语"}},
                {"ct", new[] {"Yue Chinese", "粤语"}},
                {"hi", new[] {"Hindi", "印地语"}},
                {"ur", new[] {"Urdu", "乌尔都语"}},
                {"ms", new[] {"Malay", "马来语"}},
                {"uz", new[] {"Uzbek", "乌兹别克语"}},
                {"ar", new[] {"Arabic", "阿拉伯语"}},
                {"fa", new[] {"Persian", "波斯语"}},
                {"bn", new[] {"Bengali", "孟加拉语"}},
                {"ta", new[] {"Tamil", "泰米尔语"}},
                {"te", new[] {"Telugu", "泰卢固语"}},
                {"ug", new[] {"Uighur", "维吾尔语"}},
                {"gu", new[] {"Gujarati", "古吉拉特语"}},
                {"my", new[] {"Burmese", "缅甸语"}},
                {"tl", new[] {"Tagalog", "塔加洛语"}},
                {"kk", new[] {"Kazakh", "哈萨克语"}},
                {"or", new[] {"Oriya / Odia", "奥里亚语"}},
                {"ne", new[] {"Nepali", "尼泊尔语"}},
                {"mn", new[] {"Mongolian", "蒙古语"}},
                {"km", new[] {"Khmer", "高棉语"}},
                {"jv", new[] {"Javanese", "爪哇语"}},
                {"lo", new[] {"Lao", "老挝语"}},
                {"si", new[] {"Sinhala", "僧伽罗语"}},
                {"fil", new[] {"Filipino", "菲律宾语"}},
                {"ps", new[] {"Pushto", "普什图语"}},
                {"pa", new[] {"Panjabi", "旁遮普语"}},
                {"kab", new[] {"Kabyle", "卡拜尔语"}},
                {"ba", new[] {"Bashkir", "巴什基尔语"}},
                {"ks", new[] {"Kashmiri", "克什米尔语"}},
                {"tg", new[] {"Tajik", "塔吉克语"}},
                {"su", new[] {"Sundanese", "巽他语"}},
                {"mr", new[] {"Marathi", "马拉地语"}},
                {"ky", new[] {"Kirghiz", "吉尔吉斯语"}},
                {"az", new[] {"Azerbaijani", "阿塞拜疆语"}}
            },

                // 初始化语言区域代码字典
                LanguageRegionCodes = new Dictionary<string, string[]>
            {
                {"zh-CN", new[] {"Chinese (Mandarin)", "中文(普通话)"}},
                {"zh-TW", new[] {"Chinese (Taiwan)", "中文(台湾)"}},
                {"zh-WU", new[] {"Chinese (Wuyu)", "中文(吴语)"}},
                {"zh-SICHUAN", new[] {"Chinese (Sichuan)", "中文(四川话)"}},
                {"zh-SHANXI", new[] {"Chinese (Shanxi)", "中文(山西话)"}},
                {"zh-ANHUI", new[] {"Chinese (Anhui)", "中文(安徽话)"}},
                {"zh-TIANJIN", new[] {"Chinese (Tianjin)", "中文(天津话)"}},
                {"zh-NINGXIA", new[] {"Chinese (Ningxia)", "中文(宁夏话)"}},
                {"zh-SHAANXI", new[] {"Chinese (Shanxi)", "中文(陕西话)"}},
                {"zh-HEBEI", new[] {"Chinese (Hebei)", "中文(河北话)"}},
                {"zh-SHANDONG", new[] {"Chinese (Shandong)", "中文(山东话)"}},
                {"zh-GUANGDONG", new[] {"Chinese (Guangdong)", "中文(广东话)"}},
                {"zh-SHANGHAI", new[] {"Chinese (Shanghai)", "中文(上海话)"}},
                {"zh-HUBEI", new[] {"Chinese (Hubei)", "中文(湖北话)"}},
                {"zh-LIAONING", new[] {"Chinese (Liaoning)", "中文(辽宁话)"}},
                {"zh-GANSU", new[] {"Chinese (Gansu)", "中文(甘肃话)"}},
                {"zh-FUJIAN", new[] {"Chinese (Fujian)", "中文(福建话)"}},
                {"zh-HUNAN", new[] {"Chinese (Hunan)", "中文(湖南话)"}},
                {"zh-HENAN", new[] {"Chinese (Henan)", "中文(河南话)"}},
                {"zh-YUNNAN", new[] {"Chinese (Yunnan)", "中文(云南话)"}},
                {"zh-MINNAN", new[] {"Chinese (Minnan)", "中文(闽南语)"}},
                {"zh-WENZHOU", new[] {"Chinese (Wenzhou)", "中文(温州话)"}},
                {"ja-JP", new[] {"Japanese", "日语"}},
                {"th-TH", new[] {"Thai", "泰语"}},
                {"ru-RU", new[] {"Russian", "俄语"}},
                {"ko-KR", new[] {"Korean", "韩语"}},
                {"id-ID", new[] {"Indonesian", "印度尼西亚语"}},
                {"vi-VN", new[] {"Vietnamese", "越南语"}},
                {"ct-NULL", new[] {"Yue (Chinese)", "粤语"}},
                {"ct-HK", new[] {"Yue (Hongkong)", "粤语(香港)"}},
                {"ct-GZ", new[] {"Yue (Guangdong)", "粤语(广东)"}},
                {"hi-IN", new[] {"Hindi", "印地语"}},
                {"ur-IN", new[] {"Urdu", "乌尔都语(印度)"}},
                {"ur-PK", new[] {"Urdu (Islamic Republic of Pakistan)", "乌尔都语"}},
                {"ms-MY", new[] {"Malay", "马来语"}},
                {"uz-UZ", new[] {"Uzbek", "乌兹别克语"}},
                {"ar-MA", new[] {"Arabic (Morocco)", "阿拉伯语(摩洛哥)"}},
                {"ar-GLA", new[] {"Arabic", "阿拉伯语"}},
                {"ar-SA", new[] {"Arabic (Saudi Arabia)", "阿拉伯语(沙特)"}},
                {"ar-EG", new[] {"Arabic (Egypt)", "阿拉伯语(埃及)"}},
                {"ar-KW", new[] {"Arabic (Kuwait)", "阿拉伯语(科威特)"}},
                {"ar-LY", new[] {"Arabic (Libya)", "阿拉伯语(利比亚)"}},
                {"ar-JO", new[] {"Arabic (Jordan)", "阿拉伯语(约旦)"}},
                {"ar-AE", new[] {"Arabic (U.A.E.)", "阿拉伯语(阿联酋)"}},
                {"ar-LVT", new[] {"Arabic (Levant)", "阿拉伯语(黎凡特)"}},
                {"fa-IR", new[] {"Persian", "波斯语"}},
                {"bn-BD", new[] {"Bengali", "孟加拉语"}},
                {"ta-SG", new[] {"Tamil (Singaporean)", "泰米尔语(新加坡)"}},
                {"ta-LK", new[] {"Tamil (Sri Lankan)", "泰米尔语(斯里兰卡)"}},
                {"ta-IN", new[] {"Tamil (India)", "泰米尔语(印度)"}},
                {"ta-MY", new[] {"Tamil (Malaysia)", "泰米尔语(马来西亚)"}},
                {"te-IN", new[] {"Telugu", "泰卢固语"}},
                {"ug-NULL", new[] {"Uighur", "维吾尔语"}},
                {"ug-CN", new[] {"Uighur", "维吾尔语"}},
                {"gu-IN", new[] {"Gujarati", "古吉拉特语"}},
                {"my-MM", new[] {"Burmese", "缅甸语"}},
                {"tl-PH", new[] {"Tagalog", "塔加洛语"}},
                {"kk-KZ", new[] {"Kazakh", "哈萨克语"}},
                {"or-IN", new[] {"Oriya / Odia", "奥里亚语"}},
                {"ne-NP", new[] {"Nepali", "尼泊尔语"}},
                {"mn-MN", new[] {"Mongolian", "蒙古语"}},
                {"km-KH", new[] {"Khmer", "高棉语"}},
                {"jv-ID", new[] {"Javanese", "爪哇语"}},
                {"lo-LA", new[] {"Lao", "老挝语"}},
                {"si-LK", new[] {"Sinhala", "僧伽罗语"}},
                {"fil-PH", new[] {"Filipino", "菲律宾语"}},
                {"ps-AF", new[] {"Pushto", "普什图语"}},
                {"pa-IN", new[] {"Panjabi", "旁遮普语"}},
                {"kab-NULL", new[] {"Kabyle", "卡拜尔语"}},
                {"ba-NULL", new[] {"Bashkir", "巴什基尔语"}},
                {"ks-IN", new[] {"Kashmiri", "克什米尔语"}},
                {"tg-TJ", new[] {"Tajik", "塔吉克语"}},
                {"su-ID", new[] {"Sundanese", "巽他语"}},
                {"mr-IN", new[] {"Marathi", "马拉地语"}},
                {"ky-KG", new[] {"Kirghiz", "吉尔吉斯语"}},
                {"az-AZ", new[] {"Azerbaijani", "阿塞拜疆语"}}
            }
            };
        }

        /// <summary>
        /// 示例：根据语言简码获取中文名称
        /// </summary>
        /// <param name="langCode">语言简码（如 zh/ja）</param>
        /// <returns>中文名称，不存在则返回空</returns>
        public string GetLanguageChineseName(string langCode)
        {
            if (LanguageCodes.TryGetValue(langCode, out var names))
            {
                return names[1];
            }
            return string.Empty;
        }

        /// <summary>
        /// 示例：根据语言-区域简码获取中文名称
        /// </summary>
        /// <param name="langRegionCode">语言-区域简码（如 zh-CN/ct-HK）</param>
        /// <returns>中文名称，不存在则返回空</returns>
        public string GetLanguageRegionChineseName(string langRegionCode)
        {
            if (LanguageRegionCodes.TryGetValue(langRegionCode, out var names))
            {
                return names[1];
            }
            return string.Empty;
        }
    }
}