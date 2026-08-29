"""
Synthetic (non-PII) resume / job-description test pairs for evaluating the
Resume Suite scoring pipeline. Covers same-field strong matches, cross-field
mismatches, a partial-seniority match, and an experience-gap case, so the
resulting score distribution isn't clustered at the top.

Nothing here is a real person's resume — all content is representative/
generic, written for testing purposes only.
"""

TEST_PAIRS = [
    {
        "id": "P01_swe_good",
        "label": "Software Engineer vs matching JD",
        "field": "Software Engineering",
        "resume_text": """
Aditi Rao
Software Engineer

Summary: Backend-focused software engineer with 3 years of experience building
and shipping production web services.

Experience:
Software Engineer, Bright Cart Technologies (2023-2026)
- Built and maintained REST APIs in Python (Flask, FastAPI) serving 200k+ daily requests.
- Migrated a monolithic order-processing service to microservices deployed on AWS (EC2, Lambda, S3).
- Wrote integration tests, set up CI/CD pipelines using GitHub Actions.
- Collaborated in an Agile team, ran sprint planning and code reviews.

Software Engineering Intern, Loop Systems (2022)
- Built a React dashboard consuming an internal REST API.
- Worked with a small team using Git and Jira for sprint tracking.

Skills: Python, Java, REST APIs, AWS (EC2, Lambda, S3), Git, Docker, SQL (PostgreSQL),
React, Agile/Scrum, CI/CD, unit testing.

Education: B.S. Computer Science, National Institute of Technology (2022).
""",
        "jd_text": """
Software Engineer — Backend

We're looking for a Software Engineer to join our platform team.

Responsibilities:
- Design and build REST APIs and backend services in Python.
- Deploy and maintain services on AWS or GCP.
- Write tests, participate in code review, and contribute to CI/CD pipelines.
- Work in an Agile team with weekly sprints.

Requirements:
- 2-4 years of professional software engineering experience.
- Strong Python skills; familiarity with Flask/FastAPI or Django.
- Experience with cloud platforms (AWS or GCP) and Git.
- Bachelor's degree in Computer Science or related field.
- Nice to have: Docker, Kubernetes, React.
""",
    },
    {
        "id": "P02_swe_vs_data_analyst",
        "label": "Software Engineer resume vs Data Analyst JD",
        "field": "Cross-field (SWE -> Data)",
        "resume_text": """
Aditi Rao
Software Engineer

Summary: Backend-focused software engineer with 3 years of experience building
and shipping production web services.

Experience:
Software Engineer, Bright Cart Technologies (2023-2026)
- Built and maintained REST APIs in Python (Flask, FastAPI) serving 200k+ daily requests.
- Migrated a monolithic order-processing service to microservices deployed on AWS (EC2, Lambda, S3).
- Wrote integration tests, set up CI/CD pipelines using GitHub Actions.
- Collaborated in an Agile team, ran sprint planning and code reviews.

Software Engineering Intern, Loop Systems (2022)
- Built a React dashboard consuming an internal REST API.
- Worked with a small team using Git and Jira for sprint tracking.

Skills: Python, Java, REST APIs, AWS (EC2, Lambda, S3), Git, Docker, SQL (PostgreSQL),
React, Agile/Scrum, CI/CD, unit testing.

Education: B.S. Computer Science, National Institute of Technology (2022).
""",
        "jd_text": """
Data Analyst

We're hiring a Data Analyst to support product and business decisions.

Responsibilities:
- Write SQL queries to pull and analyze data from the warehouse.
- Build dashboards in Tableau or Power BI for weekly business reviews.
- Run A/B test analysis and present statistical findings to stakeholders.
- Partner with product and marketing teams to define key metrics.

Requirements:
- 2+ years in a data analyst or business analyst role.
- Strong SQL and Excel skills; Tableau or Power BI experience.
- Working knowledge of statistics (hypothesis testing, regression).
- Python (pandas) is a plus but not required.
- Excellent stakeholder communication and presentation skills.
""",
    },
    {
        "id": "P03_data_analyst_good",
        "label": "Data Analyst vs matching JD",
        "field": "Data Analytics",
        "resume_text": """
Karan Mehta
Data Analyst

Summary: Data analyst with 2 years of experience turning raw data into
dashboards and recommendations for product and marketing teams.

Experience:
Data Analyst, Northwind Retail (2024-2026)
- Wrote complex SQL queries against a Snowflake warehouse to support weekly business reviews.
- Built and maintained 12+ Tableau dashboards tracking revenue, retention, and funnel metrics.
- Designed and analyzed A/B tests for checkout flow changes, using Python (pandas, scipy) for
  statistical significance testing.
- Presented findings to marketing and product leadership monthly.

Data Analyst Intern, Northwind Retail (2023)
- Cleaned and joined datasets from multiple sources in Excel and SQL.
- Supported ad-hoc reporting requests from the finance team.

Skills: SQL (Snowflake, PostgreSQL), Python (pandas, numpy, scipy), Tableau, Excel,
A/B testing, statistics, data visualization, stakeholder communication.

Education: B.S. Statistics, Delhi University (2023).
""",
        "jd_text": """
Data Analyst

We're hiring a Data Analyst to support product and business decisions.

Responsibilities:
- Write SQL queries to pull and analyze data from the warehouse.
- Build dashboards in Tableau or Power BI for weekly business reviews.
- Run A/B test analysis and present statistical findings to stakeholders.
- Partner with product and marketing teams to define key metrics.

Requirements:
- 2+ years in a data analyst or business analyst role.
- Strong SQL and Excel skills; Tableau or Power BI experience.
- Working knowledge of statistics (hypothesis testing, regression).
- Python (pandas) is a plus but not required.
- Excellent stakeholder communication and presentation skills.
""",
    },
    {
        "id": "P04_data_analyst_vs_marketing_manager",
        "label": "Data Analyst resume vs Marketing Manager JD",
        "field": "Cross-field (Data -> Marketing Mgmt)",
        "resume_text": """
Karan Mehta
Data Analyst

Summary: Data analyst with 2 years of experience turning raw data into
dashboards and recommendations for product and marketing teams.

Experience:
Data Analyst, Northwind Retail (2024-2026)
- Wrote complex SQL queries against a Snowflake warehouse to support weekly business reviews.
- Built and maintained 12+ Tableau dashboards tracking revenue, retention, and funnel metrics.
- Designed and analyzed A/B tests for checkout flow changes, using Python (pandas, scipy) for
  statistical significance testing.
- Presented findings to marketing and product leadership monthly.

Data Analyst Intern, Northwind Retail (2023)
- Cleaned and joined datasets from multiple sources in Excel and SQL.
- Supported ad-hoc reporting requests from the finance team.

Skills: SQL (Snowflake, PostgreSQL), Python (pandas, numpy, scipy), Tableau, Excel,
A/B testing, statistics, data visualization, stakeholder communication.

Education: B.S. Statistics, Delhi University (2023).
""",
        "jd_text": """
Marketing Manager

We're looking for a Marketing Manager to own brand strategy and campaign execution.

Responsibilities:
- Define and execute quarterly marketing campaigns across channels.
- Own the social media content calendar and brand voice.
- Manage a $500k annual marketing budget and vendor relationships.
- Lead a team of 3 marketing coordinators.

Requirements:
- 5+ years of marketing experience, 2+ years in a management role.
- Proven track record running integrated campaigns (paid, social, email).
- Strong budget management and team leadership skills.
- Experience with brand positioning and go-to-market strategy.
""",
    },
    {
        "id": "P05_digital_marketing_good",
        "label": "Digital Marketing Specialist vs matching JD",
        "field": "Digital Marketing",
        "resume_text": """
Priya Nair
Digital Marketing Specialist

Summary: Digital marketer with 3 years of experience running paid search,
paid social, and SEO programs for D2C brands.

Experience:
Digital Marketing Specialist, Verve Commerce (2023-2026)
- Managed Google Ads and Meta Ads campaigns with a combined monthly budget of $40k.
- Improved organic search traffic by 65% over 18 months through on-page SEO and content strategy.
- Built and monitored GA4 dashboards to track campaign ROI and funnel conversion.
- Ran email marketing campaigns (Klaviyo) with 22% average open rate.

Marketing Coordinator, Verve Commerce (2022-2023)
- Assisted with content calendar planning and social media scheduling.
- Coordinated with design team on ad creative production.

Skills: Google Ads, Meta Ads, SEO/SEM, GA4, Klaviyo, content strategy,
campaign optimization, A/B testing of ad creative.

Education: B.A. Marketing, Symbiosis Institute (2022).
""",
        "jd_text": """
Digital Marketing Specialist

We need a Digital Marketing Specialist to grow paid and organic acquisition.

Responsibilities:
- Plan and run paid search and paid social campaigns (Google Ads, Meta Ads).
- Own SEO strategy: keyword research, on-page optimization, content briefs.
- Track performance in GA4 and report on CAC, ROAS, and funnel conversion.
- Manage the content calendar across email and social channels.

Requirements:
- 2-4 years of hands-on digital marketing experience.
- Proficiency with Google Ads, Meta Ads Manager, and GA4.
- Working knowledge of SEO best practices.
- Comfortable analyzing campaign data and presenting recommendations.
""",
    },
    {
        "id": "P06_mech_eng_good",
        "label": "Mechanical Engineer vs matching JD",
        "field": "Mechanical Engineering",
        "resume_text": """
Rahul Verma
Mechanical Design Engineer

Summary: Mechanical engineer with 4 years of experience in automotive
component design, from concept through manufacturing handoff.

Experience:
Mechanical Design Engineer, Torque Auto Components (2022-2026)
- Designed sheet-metal and injection-molded parts in SolidWorks for interior trim assemblies.
- Ran FEA simulations (Ansys) to validate structural and thermal performance.
- Applied GD&T on production drawings; coordinated with suppliers on DFM feedback.
- Led design reviews with cross-functional manufacturing and quality teams.

Design Engineer Intern, Torque Auto Components (2021)
- Supported CAD modeling and drawing updates for legacy tooling.

Skills: SolidWorks, AutoCAD, Ansys FEA, GD&T, DFM/DFA, sheet metal design,
injection molding, manufacturing processes, project coordination.

Education: B.Tech Mechanical Engineering, VIT (2021).
""",
        "jd_text": """
Mechanical Design Engineer

We're hiring a Mechanical Design Engineer for our product design team.

Responsibilities:
- Design mechanical components and assemblies in SolidWorks or AutoCAD.
- Run FEA to validate designs against structural and thermal requirements.
- Apply GD&T and prepare production-ready drawings.
- Work with manufacturing partners on DFM/DFA feedback.

Requirements:
- 3-5 years of mechanical design experience, ideally in automotive or consumer products.
- Strong CAD skills (SolidWorks or AutoCAD) and FEA experience.
- Solid understanding of GD&T and manufacturing processes.
- B.Tech/B.E. in Mechanical Engineering.
""",
    },
    {
        "id": "P07_mech_eng_vs_swe",
        "label": "Mechanical Engineer resume vs Software Engineer JD",
        "field": "Cross-field (Mech -> SWE)",
        "resume_text": """
Rahul Verma
Mechanical Design Engineer

Summary: Mechanical engineer with 4 years of experience in automotive
component design, from concept through manufacturing handoff.

Experience:
Mechanical Design Engineer, Torque Auto Components (2022-2026)
- Designed sheet-metal and injection-molded parts in SolidWorks for interior trim assemblies.
- Ran FEA simulations (Ansys) to validate structural and thermal performance.
- Applied GD&T on production drawings; coordinated with suppliers on DFM feedback.
- Led design reviews with cross-functional manufacturing and quality teams.

Design Engineer Intern, Torque Auto Components (2021)
- Supported CAD modeling and drawing updates for legacy tooling.

Skills: SolidWorks, AutoCAD, Ansys FEA, GD&T, DFM/DFA, sheet metal design,
injection molding, manufacturing processes, project coordination.

Education: B.Tech Mechanical Engineering, VIT (2021).
""",
        "jd_text": """
Software Engineer — Backend

We're looking for a Software Engineer to join our platform team.

Responsibilities:
- Design and build REST APIs and backend services in Python.
- Deploy and maintain services on AWS or GCP.
- Write tests, participate in code review, and contribute to CI/CD pipelines.
- Work in an Agile team with weekly sprints.

Requirements:
- 2-4 years of professional software engineering experience.
- Strong Python skills; familiarity with Flask/FastAPI or Django.
- Experience with cloud platforms (AWS or GCP) and Git.
- Bachelor's degree in Computer Science or related field.
- Nice to have: Docker, Kubernetes, React.
""",
    },
    {
        "id": "P08_financial_analyst_good",
        "label": "Financial Analyst vs matching JD",
        "field": "Finance",
        "resume_text": """
Sneha Kulkarni
Financial Analyst

Summary: Financial analyst with 3 years of experience in FP&A, supporting
budgeting, forecasting, and variance analysis for a mid-size company.

Experience:
Financial Analyst, Meridian Industries (2023-2026)
- Built and maintained a 3-statement financial model used for quarterly forecasting.
- Prepared monthly variance analysis comparing actuals to budget, presented to department heads.
- Automated recurring reporting using Excel (Power Query) and SQL queries against the ERP database.
- Supported annual budgeting process across 6 business units.

Finance Rotational Associate, Meridian Industries (2022-2023)
- Rotated through AP, AR, and treasury functions.
- Assisted with month-end close and journal entry review.

Skills: Financial modeling, Excel (advanced, Power Query), forecasting, variance analysis,
SQL, budgeting, presentations to leadership, CFA Level 1 candidate.

Education: B.Com Finance, Narsee Monjee Institute (2022).
""",
        "jd_text": """
Financial Analyst

We're hiring a Financial Analyst to join our FP&A team.

Responsibilities:
- Build and maintain financial models for forecasting and scenario planning.
- Prepare monthly variance analysis and present to department leadership.
- Support the annual budgeting cycle across business units.
- Partner with accounting on month-end close reporting.

Requirements:
- 2-4 years of FP&A or financial analysis experience.
- Strong Excel skills; SQL experience is a plus.
- Ability to communicate financial findings clearly to non-finance stakeholders.
- B.Com/BBA in Finance or related field; CFA candidacy a plus.
""",
    },
    {
        "id": "P09_graphic_designer_good",
        "label": "Graphic Designer vs matching JD",
        "field": "Graphic Design",
        "resume_text": """
Ananya Iyer
Graphic Designer

Summary: Graphic designer with 3 years of agency experience across brand
identity, print, and digital design.

Experience:
Graphic Designer, Studio Ember (2023-2026)
- Led brand identity projects for 8+ clients: logo design, typography systems, brand guidelines.
- Designed digital ad creative and social media assets in Photoshop and Illustrator.
- Created UI mockups and clickable prototypes in Figma for two client web projects.
- Presented design concepts directly to clients and incorporated feedback.

Junior Designer, Studio Ember (2022-2023)
- Supported print production: brochures, packaging mockups, trade show materials.

Skills: Adobe Photoshop, Illustrator, InDesign, Figma, typography, branding,
print and digital design, client presentations.

Education: B.Des Visual Communication, MIT Institute of Design (2022).
""",
        "jd_text": """
Graphic Designer

We're looking for a Graphic Designer to support brand and marketing design needs.

Responsibilities:
- Design brand identity assets: logos, typography systems, style guides.
- Produce digital ad creative and social media graphics.
- Create UI mockups in Figma for marketing landing pages.
- Present design work to internal stakeholders and clients.

Requirements:
- 2-4 years of graphic design experience, agency background a plus.
- Strong Adobe Creative Suite skills (Photoshop, Illustrator, InDesign).
- Working knowledge of Figma.
- Strong portfolio demonstrating branding and digital design work.
""",
    },
    {
        "id": "P10_hr_generalist_partial",
        "label": "HR Generalist resume vs HR Business Partner (senior) JD",
        "field": "Human Resources (seniority gap)",
        "resume_text": """
Meera Iyer
HR Generalist

Summary: HR generalist with 3 years of experience across recruitment,
onboarding, and employee relations for a 200-person tech company.

Experience:
HR Generalist, Falcon Softworks (2023-2026)
- Managed end-to-end recruitment for engineering and operations roles (40+ hires/year).
- Ran new-hire onboarding programs and maintained HRIS records in Workday.
- Handled first-line employee relations issues, escalating complex cases to HR leadership.
- Coordinated with payroll on offer letters, compensation changes, and exit processing.

HR Coordinator, Falcon Softworks (2022-2023)
- Scheduled interviews, maintained applicant tracking system, supported new-hire paperwork.

Skills: Recruitment, onboarding, employee relations, Workday (HRIS), payroll coordination,
applicant tracking systems, HR compliance basics.

Education: B.A. Human Resource Management, Christ University (2022).
""",
        "jd_text": """
HR Business Partner

We're hiring a senior HR Business Partner to work directly with executive leadership.

Responsibilities:
- Partner with senior leaders on org design, workforce planning, and succession planning.
- Lead compensation strategy and calibration for a 500+ person org.
- Coach executives through complex organizational change and restructuring.
- Own the annual talent review and leadership development strategy.

Requirements:
- 6+ years of HR experience, including 3+ years in a strategic HRBP role.
- Proven experience advising senior/executive stakeholders.
- Deep knowledge of compensation strategy and organizational design.
- Change management certification a plus.
""",
    },
    {
        "id": "P11_civil_eng_good",
        "label": "Civil Engineer vs matching JD",
        "field": "Civil Engineering",
        "resume_text": """
Arjun Nair
Civil Engineer

Summary: Civil engineer with 4 years of experience in site development and
infrastructure design for residential and commercial projects.

Experience:
Civil Engineer, Skyline Infra Consultants (2022-2026)
- Designed site grading, drainage, and utility layouts in AutoCAD Civil 3D for 10+ projects.
- Performed structural analysis for retaining walls and foundation systems.
- Managed project timelines and coordinated with contractors during construction.
- Currently completing requirements for Professional Engineer (PE) licensure.

Civil Engineer Intern, Skyline Infra Consultants (2021)
- Assisted with site surveys and preliminary grading plans.

Skills: AutoCAD Civil 3D, structural analysis, site planning, drainage design,
project management, construction coordination, PE license (in progress).

Education: B.Tech Civil Engineering, Anna University (2021).
""",
        "jd_text": """
Civil Engineer

We're hiring a Civil Engineer to support site development and infrastructure projects.

Responsibilities:
- Design site grading, drainage, and utility plans using AutoCAD Civil 3D.
- Perform structural analysis for foundations and retaining structures.
- Coordinate with contractors and manage project timelines during construction.

Requirements:
- 3-6 years of civil engineering experience, site development focus preferred.
- Strong AutoCAD Civil 3D skills.
- PE license or actively pursuing licensure preferred.
- B.Tech/B.E. in Civil Engineering.
""",
    },
    {
        "id": "P12_fresher_vs_senior",
        "label": "Fresh graduate resume vs Senior Software Engineer JD",
        "field": "Experience-gap case",
        "resume_text": """
Devansh Kapoor
Recent Graduate — Computer Science

Summary: Recent computer science graduate with internship experience and
strong academic project work, seeking an entry-level software role.

Experience:
Software Engineering Intern, Clearline Labs (Jan 2026 - Jun 2026)
- Fixed bugs and added small features to an internal Python Flask application.
- Wrote unit tests under supervision of a senior engineer.
- Participated in daily standups and sprint planning as an observer.

Academic Projects:
- Built a course-project e-commerce site using Python/Django and PostgreSQL.
- Built a command-line task manager in Java as part of a data structures course.

Skills: Python, Java, basic SQL, Git, HTML/CSS, coursework in data structures,
algorithms, and databases.

Education: B.S. Computer Science, National Institute of Technology (2026).
""",
        "jd_text": """
Senior Software Engineer

We're hiring a Senior Software Engineer to lead design of core platform services.

Responsibilities:
- Own system design for distributed backend services handling millions of requests/day.
- Mentor junior and mid-level engineers, run design reviews.
- Drive architecture decisions for scalability and reliability.
- Partner with engineering leadership on technical roadmap.

Requirements:
- 8+ years of professional software engineering experience.
- Proven experience with distributed systems and cloud architecture (AWS/GCP).
- Track record of mentoring engineers and leading technical initiatives.
- Strong Python or Go skills.
""",
    },
]
