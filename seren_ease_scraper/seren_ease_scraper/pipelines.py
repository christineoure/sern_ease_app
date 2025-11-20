import json
import os
class SaveToDrivePipeline:
    def open_spider(self, spider):
        # Local file path for macOS
        os.makedirs("data", exist_ok=True)
        self.file_path = "data/mental_health_articles.jsonl"
        spider.logger.info(f"✓ Saving locally to: {self.file_path}")
        self.file = open(self.file_path, "w", encoding="utf-8")
        self.item_count = 0
    def close_spider(self, spider):
        self.file.close()
        spider.logger.info(f"✓ Finished! Saved {self.item_count} items to: {self.file_path}")
    def process_item(self, item, spider):
        self.file.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        self.file.flush()
        self.item_count += 1
        return item