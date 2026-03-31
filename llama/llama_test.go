package llama

import (
	"bufio"
	"bytes"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

// https://github.com/ollama/ollama/issues/7978
const issue7978JSONSchema = `{
  "type": "object",
  "properties": {
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "explanation": { "type": "string" },
          "output": { "type": "string" },
          "nested": {
            "type": "object",
            "properties": {
              "deep": { "type": "string" }
            }
          }
        },
        "required": ["explanation", "output"],
        "additionalProperties": false
      }
    },
    "final_answer": { "type": "string" },
    "01_numbered_key": { "type": "string" },
    "numbers": {
      "type": "array",
      "items": { "type": "number" }
    },
    "booleans": {
      "type": "array", 
      "items": { "type": "boolean" }
    },
    "mixed": {
      "type": "array",
      "items": {
        "oneOf": [
          { "type": "string" },
          { "type": "number" },
          { "type": "boolean" }
        ]
      }
    }
  },
  "required": ["steps", "final_answer"],
  "additionalProperties": false
}`

func TestIssue7978(t *testing.T) {
	g := SchemaToGrammar([]byte(issue7978JSONSchema))
	if g == nil {
		t.Fatal("failed to convert JSON schema to grammar")
	}

	t.Logf("grammar:\n%s", g)
	t.Log()

	var got string
	s := bufio.NewScanner(bytes.NewReader(g))
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		step, _, _ := strings.Cut(line, " ::= ")
		step = strings.TrimSpace(step)
		if step == "root" {
			got = line
		}
	}

	want := `root ::= "{" space steps-kv "," space final-answer-kv ( "," space ( 01-numbered-key-kv 01-numbered-key-rest | numbers-kv numbers-rest | booleans-kv booleans-rest | mixed-kv ) )? "}" space`
	if got != want {
		t.Errorf("root =\n%qwant:\n%q", got, want)
	}
}

func TestSchemaToGrammar(t *testing.T) {
	cases := []struct {
		schema string
		prefix []byte // nil is checked as nil
	}{
		{`invalid`, nil},

		// Simple heuristic/smoke test
		{`{"type":"object"}`, []byte("root ::= object")},
	}

	for _, c := range cases {
		t.Run(c.schema, func(t *testing.T) {
			g := SchemaToGrammar([]byte(c.schema))
			if c.prefix == nil && g != nil {
				t.Fatalf("grammar = %v, want nil", g)
			}
			if !bytes.HasPrefix(g, c.prefix) {
				t.Errorf("grammar = %q, want %q", g, c.prefix)
			}
		})
	}
}

func TestNewContextParamsRejectsTurboQuant(t *testing.T) {
	for _, cacheType := range []string{"tq25", "tq35", "tq3", "tq4"} {
		if _, err := NewContextParams(128, 16, 1, 1, ml.FlashAttentionDisabled, cacheType); err == nil {
			t.Fatalf("NewContextParams(..., %q) error = nil, want rejection", cacheType)
		} else {
			if !strings.Contains(err.Error(), "legacy llama runner") {
				t.Fatalf("NewContextParams(..., %q) error = %q, want legacy llama runner message", cacheType, err)
			}
			if cacheType == "tq3" || cacheType == "tq4" {
				if !strings.Contains(err.Error(), "tq35") {
					t.Fatalf("NewContextParams(..., %q) error = %q, want normalized tq35", cacheType, err)
				}
			}
		}
	}
}
