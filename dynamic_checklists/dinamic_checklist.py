import abc
import re
import json
import requests
import time

class BaseChecklist(abc.ABC):

    @staticmethod
    def get_checklist(*args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class ScholarCitationChecklist(BaseChecklist):

    @staticmethod
    def get_checklist(*args, **kwargs):

        url = kwargs.get("url")
        author_name = kwargs.get("name")
        
        include_i10_index = kwargs.get("include_i10_index", True)
        include_h_index = kwargs.get("include_h_index", True)
        
        max_tries=3
        
        for i in range(max_tries):
            try:
                serper_scrape_endpoint = "https://scrape.serper.dev"

                payload = json.dumps({"url": url})
                headers = {
                    "X-API-KEY": "9ef3a626ae63a9990f924fc443c1ed3525bf1cdf",
                    "Content-Type": "application/json",
                }

                print( url)

                response = requests.request("POST", serper_scrape_endpoint, headers=headers, data=payload)

                response_data = response.json()
                
                text=response_data.get("text")
                
                
                citations = re.findall(r"Citations\s+(\d+)", text)[0]
                h_index = re.findall(r"h-index\s+(\d+)", text)[0]
                i10_index = re.findall(r"i10-index\s+(\d+)", text)[0]
                
                
                checklist=[
                    f"The report must mention that the author {author_name} has {citations} citations.",
                ]
                if include_h_index:
                    checklist.append(f"The report must mention that the author {author_name} has an h-index of {h_index}.")
                if include_i10_index:
                    checklist.append(f"The report must mention that the author {author_name} has an i10-index of {i10_index}.")
                
                return checklist
            except Exception as e:
                print(f"Error getting scholar citation checklist: {e}")
                if i < max_tries - 1:
                    print(f"Retrying... ({i+1}/{max_tries})")
                    time.sleep(1)
                else:
                    raise e



available_dynamic_checklists = {
    "ScholarCitationChecklist": ScholarCitationChecklist,
}