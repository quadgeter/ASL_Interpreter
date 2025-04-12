import pkg_resources

# Get all installed packages in the current environment
installed_packages = pkg_resources.working_set
requirements = sorted(["{}=={}".format(pkg.key, pkg.version) for pkg in installed_packages])

# Save to requirements.txt
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements))

