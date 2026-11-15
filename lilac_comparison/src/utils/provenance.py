def experiment_sources(project_root):
    paths = set()

    for name in ("src", "data", "configs", "audit"):
        directory = project_root / name
        paths.update(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )

    return sorted(str(path.relative_to(project_root)) for path in paths)
