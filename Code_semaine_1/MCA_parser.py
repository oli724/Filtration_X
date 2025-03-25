class MCA:
    def __init__(self, file_path):
        self.sections = {}  # Store sections dynamically
        self._parse_file(file_path)

    def _parse_file(self, file_path):
        current_section = None

        with open(file_path, "r", encoding="latin-1") as file:
            for line in file:
                line = line.strip()

                # Detect section headers (e.g., <<DATA>>, <<END>>)
                if line.startswith("<<") and line.endswith(">>"):
                    section_name = line[2:-2]  # Extract section name

                    # Ignore sections ending with "END>>"
                    if section_name.endswith("END"):
                        current_section = None
                        continue

                    # Start a new section
                    current_section = section_name
                    self.sections[current_section] = []  # Initialize empty list
                    continue

                # Store data in the corresponding section
                if current_section:
                    self.sections[current_section].append(line)

        # Process specific sections (e.g., convert DATA to integers)
        if "DATA" in self.sections:
            self.sections["DATA"] = [int(x) for x in self.sections["DATA"] if x.isdigit()]

    def __getattr__(self, name):
        if name in self.sections:
            return self.sections[name]
        raise AttributeError(f"'MCA' object has no attribute '{name}'")

