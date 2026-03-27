import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import useDebouncedValue from "./useDebouncedValue";

function DebouncedValueHarness({ delayMs, value }) {
  const debouncedValue = useDebouncedValue(value, delayMs);
  return <output data-testid="debounced-value">{debouncedValue}</output>;
}

describe("useDebouncedValue", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it("updates only after the debounce delay", async () => {
    const { rerender } = render(
      <DebouncedValueHarness delayMs={300} value="alpha" />,
    );

    rerender(<DebouncedValueHarness delayMs={300} value="beta" />);

    expect(screen.getByTestId("debounced-value")).toHaveTextContent("alpha");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(299);
    });
    expect(screen.getByTestId("debounced-value")).toHaveTextContent("alpha");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(screen.getByTestId("debounced-value")).toHaveTextContent("beta");
  });

  it("cancels the prior pending update when a new value arrives quickly", async () => {
    const { rerender } = render(
      <DebouncedValueHarness delayMs={200} value="first" />,
    );

    rerender(<DebouncedValueHarness delayMs={200} value="second" />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });
    rerender(<DebouncedValueHarness delayMs={200} value="third" />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });
    expect(screen.getByTestId("debounced-value")).toHaveTextContent("first");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });
    expect(screen.getByTestId("debounced-value")).toHaveTextContent("third");
  });
});
