"""Builds data/us_state_context.csv and data/us_states.geojson (run once; the app only reads the outputs).

Source: NCI/CDC State Cancer Profiles (https://statecancerprofiles.cancer.gov/), as compiled in the MIT-licensed repo
https://github.com/sergeevaleeza/cancer-data-visualization-america  (clone it and pass its path as argv[1]).
 - breast cancer (female) incidence + mortality, age-adjusted rate per 100,000, 2015
 - risk factors: 2016 BRFSS (obesity 20+, smoking 18+, no physical activity, daily fruit / vegetables)
 - census: 2012-2016 ACS (uninsured <65, below 150% poverty, median family income, bachelor's degree 25+)
Geometry: US Census Bureau 2010 cartographic boundary file (1:20m).
"""
import json, sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "cancer-data-visualization-america"
rd = lambda f: pd.read_csv(f"{src}/data/{f}.csv", encoding="utf-8-sig").rename(columns=lambda c: c.strip())
num = lambda s: pd.to_numeric(s.astype(str).str.strip().replace({"": None}), errors="coerce")
inc, mor, rk, ce = rd("incidence"), rd("mortality"), rd("risk"), rd("census")
t = pd.DataFrame({"state": inc.State, "abbr": inc.State_code, "inc": num(inc.breast_rate), "inc_n": num(inc.breast_count)})
t = t.merge(pd.DataFrame({"state": mor.State, "mort": num(mor.breast_rate), "mort_n": num(mor.breast_count)}), on="state")
t = t.merge(pd.DataFrame({"state": rk.State, "obese": num(rk.obese_BMI_20_plus_percent), "smoke": num(rk.smokers_percent),
                          "inactive": num(rk.no_physical_activity_percent), "fruit": num(rk.fruit_percent),
                          "veg": num(rk.veggies_percent)}), on="state")
t = t.merge(pd.DataFrame({"state": ce.State, "unins": num(ce.uninsured_percent), "pov": num(ce.below_150_poverty_percent),
                          "income": num(ce.median_family_income_dollars), "bach": num(ce.education_bachelors_percent)}), on="state")
t = t[~t.state.isin(["United States", "Puerto Rico"])].sort_values("state")
t.to_csv("data/us_state_context.csv", index=False)
g = json.load(open(f"{src}/data/geo_states.json"))
g["features"] = [f for f in g["features"] if f["properties"]["NAME"] != "Puerto Rico"]
for f in g["features"]:
    f["properties"] = {"NAME": f["properties"]["NAME"]}
json.dump(g, open("data/us_states.geojson", "w"), separators=(",", ":"))
print(len(t), "states written")
