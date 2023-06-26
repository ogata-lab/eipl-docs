MKDOCS_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SITE_DIR    := $(MKDOCS_ROOT)/site
CNAME       := ogata-lab.github.io

build:
	mkdocs build --config-file $(MKDOCS_ROOT)/top/mkdocs.yml  --site-dir $(SITE_DIR)
	mkdocs build --config-file $(MKDOCS_ROOT)/ja/mkdocs.yml   --site-dir $(SITE_DIR)/ja/
	mkdocs build --config-file $(MKDOCS_ROOT)/en/mkdocs.yml   --site-dir $(SITE_DIR)/en/
	cp -r top $(SITE_DIR)
publish:
	ghp-import -c $(CNAME) -r origin -b gh-pages -p site

clean:
	rm -rf  $(SITE_DIR)
