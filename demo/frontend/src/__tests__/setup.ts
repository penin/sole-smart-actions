import "@testing-library/jest-dom/vitest";

// jsdom has no layout engine; Recharts' ResponsiveContainer needs ResizeObserver to exist.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver ??= ResizeObserverStub;
