import scrapy

class MentalHealthSpider(scrapy.Spider):
    name = "mental_health"
    allowed_domains = [
        "verywellmind.com", "nami.org", "apa.org", "psychcentral.com",
        "healthdirect.gov.au", "medicalnewstoday.com",
        "mayoclinic.org", "webmd.com", "psycom.net", "mentalhealth.gov",
        "mentalhealth.org.uk", "cdc.gov", "nimh.nih.gov", "healthline.com",
        "psychologytoday.com", "blackdoginstitute.org.au",
        "anxiety.org", "depression.org", "schizophrenia.com",
        "mind.org.uk", "beyondblue.org.au", "samhsa.gov", 
        "mhanational.org", "thetrevorproject.org", "jedfoundation.org", 
        "childmind.org", "goodtherapy.org", "verywellhealth.com",
        "health.harvard.edu", "talkspace.com", "betterhelp.com",
        "healthygamer.gg", "therapistaid.com", "kff.org",
        "mentalhelp.net", "additudemag.com", "eatingdisorderhope.com", 
        "adaa.org", "dbsalliance.org", "narcotics.com", "al-anon.org",
        "psychiatry.org", "nhs.uk", "rethink.org",
        "mentalhealthamerica.net", "mentalhealthcommission.gov.au" 
    ]
    start_urls = [
        
        "https://www.verywellmind.com/mental-health-4157199",
        "https://www.nami.org/About-Mental-Illness",
        "https://www.apa.org/topics",
        "https://psychcentral.com/health",
        "https://www.healthdirect.gov.au/mental-health",
        "https://www.medicalnewstoday.com/categories/mental-health",
        "https://www.mayoclinic.org/diseases-conditions/mental-health",
        "https://www.webmd.com/mental-health/mental-health-center",
        "https://www.psycom.net/mental-health",
        "https://www.mentalhealth.gov/",
        "https://www.mentalhealth.org.uk/",
        "https://www.cdc.gov/mentalhealth",
        "https://www.nimh.nih.gov/health",
        "https://www.healthline.com/mental-health",
        "https://www.psychologytoday.com/us/topics/mental-health",
        "https://www.blackdoginstitute.org.au/",
        "https://www.anxiety.org/",
        "https://www.depression.org/",
        "https://www.schizophrenia.com/",
        "https://www.mind.org.uk/information-support/",
        "https://www.beyondblue.org.au/get-support/national-help-lines-and-websites",
        "https://www.samhsa.gov/find-help/national-helpline",
        "https://www.mhanational.org/finding-help",
        "https://www.thetrevorproject.org/get-help/",
        "https://www.jedfoundation.org/resources/",
        "https://www.childmind.org/topics/",
        "https://www.goodtherapy.org/learn-about-therapy/issues",
        "https://www.verywellhealth.com/mental-health-4160411",
        "https://www.health.harvard.edu/topics/mental-health",
        "https://www.talkspace.com/mental-health",
        "https://www.betterhelp.com/",
        "https://www.healthygamer.gg/blog",
        "https://www.therapistaid.com/therapy-worksheets/mental-health/adolescent",
        "https://www.kff.org/mental-health/",
        "https://www.mentalhelp.net/mental-illness/",
        "https://www.additudemag.com/category/adhd-add-mental-health/",
        "https://www.eatingdisorderhope.com/treatment-for-eating-disorders",
        "https://adaa.org/understanding-anxiety/resources",
        "https://www.dbsalliance.org/education/bipolar-disorder/bipolar-disorder-overview/",
        "https://www.narcotics.com/substance-abuse-treatment-types/",
        "https://al-anon.org/for-members/new-members/",
        "https://www.psychiatry.org/patients-families/what-is-mental-illness",
        "https://www.nhs.uk/mental-health/",
        "https://www.rethink.org/advice-and-information/about-mental-illness/",
        "https://www.mentalhealthamerica.net/mental-health-information",
        "https://www.mentalhealthcommission.gov.au/home",
        "https://www.nami.org/find-support/signs-symptoms",
        "https://www.nimh.nih.gov/health/topics/depression",
        "https://www.cdc.gov/mentalhealth/index.htm",
        "https://www.healthline.com/health/mental-health/types-of-mental-illness",
        "https://www.mayoclinic.org/diseases-conditions/anxiety/symptoms-causes",
        "https://www.webmd.com/schizophrenia/schizophrenia-overview",
        "https://www.psychologytoday.com/us/basics/anxiety",
        "https://www.mind.org.uk/information-support/types-of-mental-health-problems/stress/",
        "https://www.beyondblue.org.au/personal-best/pillar/wellbeing/recognising-anxiety-and-depression",
        "https://www.samhsa.gov/find-help/disaster-distress-helpline",
        "https://www.mhanational.org/conditions/bipolar-disorder",
        "https://www.thetrevorproject.org/resources/trevor-support-center/",
        "https://www.jedfoundation.org/mental-health-resource-center-2/",
        "https://www.childmind.org/article/what-is-anxiety/",
        "https://www.goodtherapy.org/learn-about-therapy/types-of-therapy",
        "https://www.verywellmind.com/common-mental-illness-diagnoses-4157198",
        "https://www.health.harvard.edu/topics/anxiety-and-stress",
        "https://www.talkspace.com/blog/topic/mental-health-advice/",
        "https://www.psycom.net/bipolar-disorder",
        "https://www.mentalhealth.gov/basics/what-is-mental-health"
        
    ]
    custom_settings = {
        'DEPTH_LIMIT': 2,
        'CLOSESPIDER_PAGECOUNT': 50,
    }

    def parse(self, response):
        """Extract content and follow article links."""
        self.logger.info(f"Parsing: {response.url}")
        
        # Extraction logic 
        title = (
            response.css("h1::text").get() or
            response.css("h2::text").get() or
            response.css("title::text").get()
        )
        paragraphs = response.css(
            "article p::text, "
            "div.article-content p::text, "
            "div.content p::text, "
            "section p::text, "
            "div p::text"
        ).getall()
        body = " ".join(p.strip() for p in paragraphs if len(p.strip()) > 30)
        
        if title and len(body) > 100:
            yield {
                "url": response.url,
                "source": response.url.split("/")[2],
                "title": title.strip(),
                "body": body[:5000],  # Limit body length
            }
            self.logger.info(f"✓ Extracted: {title.strip()[:50]}...")
        else:
            self.logger.debug(f"✗ Skipped {response.url} - insufficient content")

        # Following links logic 
        for href in response.css("a::attr(href)").getall()[:20]:
            full_url = response.urljoin(href)
            if any(domain in full_url for domain in self.allowed_domains):
                if any(keyword in full_url.lower() for keyword in [
                    "mental", "health", "wellbeing", "therapy", "stress",
                    "anxiety", "depression", "article", "disorder", "support"
                ]):
                    yield scrapy.Request(full_url, callback=self.parse)