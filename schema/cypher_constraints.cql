// Cardiometabolic graph — Neo4j constraints and indexes.
// Apply with:  cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD -f cypher_constraints.cql
// Safe to re-run: every statement uses IF NOT EXISTS.

// --- Uniqueness constraints (primary keys) ---
CREATE CONSTRAINT patient_id IF NOT EXISTS
FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE;

CREATE CONSTRAINT encounter_id IF NOT EXISTS
FOR (e:Encounter) REQUIRE e.encounter_id IS UNIQUE;

CREATE CONSTRAINT lab_id IF NOT EXISTS
FOR (l:LabResult) REQUIRE l.lab_id IS UNIQUE;

CREATE CONSTRAINT vital_id IF NOT EXISTS
FOR (v:Vital) REQUIRE v.vital_id IS UNIQUE;

CREATE CONSTRAINT rx_id IF NOT EXISTS
FOR (m:Medication) REQUIRE m.rx_id IS UNIQUE;

CREATE CONSTRAINT event_id IF NOT EXISTS
FOR (b:BehavioralEvent) REQUIRE b.event_id IS UNIQUE;

CREATE CONSTRAINT pathway_id IF NOT EXISTS
FOR (p:Pathway) REQUIRE p.pathway_id IS UNIQUE;

CREATE CONSTRAINT gene_symbol IF NOT EXISTS
FOR (g:Gene) REQUIRE g.gene_symbol IS UNIQUE;

CREATE CONSTRAINT chebi_id IF NOT EXISTS
FOR (m:Metabolite) REQUIRE m.chebi_id IS UNIQUE;

// --- Existence / required-property constraints (Enterprise only; commented out for Community) ---
// CREATE CONSTRAINT patient_birth_year IF NOT EXISTS
// FOR (p:Patient) REQUIRE p.birth_year IS NOT NULL;

// --- Secondary indexes (query-path tuning) ---
CREATE INDEX lab_patient_ts IF NOT EXISTS
FOR (l:LabResult) ON (l.patient_id, l.taken_ts);

CREATE INDEX vital_patient_ts IF NOT EXISTS
FOR (v:Vital) ON (v.patient_id, v.taken_ts);

CREATE INDEX rx_patient_start IF NOT EXISTS
FOR (m:Medication) ON (m.patient_id, m.start_ts);

CREATE INDEX behavior_patient_ts IF NOT EXISTS
FOR (b:BehavioralEvent) ON (b.patient_id, b.ts);

CREATE INDEX lab_loinc IF NOT EXISTS
FOR (l:LabResult) ON (l.loinc);

CREATE INDEX pathway_source IF NOT EXISTS
FOR (p:Pathway) ON (p.source);
