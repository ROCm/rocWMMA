# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

from rocm_docs import ROCmDocs

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'set \( VERSION_STRING\s+\"?([0-9.]+)[^0-9.]+', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"rocWMMA {version_number} Documentation"

# for PDF output on Read the Docs
project = "rocWMMA Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.setup()

external_projects_current_project = "rocwmma"

# Apply ROCmDocs sphinx variables
for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Breathe configuration - CRITICAL: Set after ROCmDocs to override defaults
# ROCmDocs may set breathe_projects but NOT breathe_default_project
breathe_projects = {"rocwmma": "doxygen/xml"}
breathe_default_project = "rocwmma"

# Force into globals to ensure Sphinx sees them
globals()['breathe_projects'] = breathe_projects
globals()['breathe_default_project'] = breathe_default_project

# Debug: Print to verify (remove this in production)
print(f"[DEBUG] breathe_projects = {breathe_projects}")
print(f"[DEBUG] breathe_default_project = {breathe_default_project}")

# Chinese localization (active)
breathe_projects = {"rocwmma": "doxygen/xml_zh"}
breathe_default_project = "rocwmma"
language = "zh_CN"
globals()['breathe_projects'] = breathe_projects
globals()['breathe_default_project'] = breathe_default_project
globals()['language'] = language

# Uncomment for Japanese localization:
# breathe_projects = {"rocwmma": "doxygen/xml_ja"}
# breathe_default_project = "rocwmma"
# language = "ja"
# globals()['breathe_projects'] = breathe_projects
# globals()['language'] = language
