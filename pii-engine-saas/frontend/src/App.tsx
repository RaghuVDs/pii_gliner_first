import React, { Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Spin } from "antd";
import AppShell from "@/components/layout/AppShell";
import ProtectedRoute from "@/routes/ProtectedRoute";

const LoginPage = React.lazy(() => import("@/pages/auth/LoginPage"));
const RegisterPage = React.lazy(() => import("@/pages/auth/RegisterPage"));
const DashboardPage = React.lazy(
  () => import("@/pages/dashboard/DashboardPage")
);
const DetectionPage = React.lazy(
  () => import("@/pages/detection/DetectionPage")
);
const TaxonomyPage = React.lazy(
  () => import("@/pages/taxonomy/TaxonomyPage")
);
const RegexRulesPage = React.lazy(
  () => import("@/pages/rules/RegexRulesPage")
);
const FieldPatternsPage = React.lazy(
  () => import("@/pages/rules/FieldPatternsPage")
);
const ContextRulesPage = React.lazy(
  () => import("@/pages/rules/ContextRulesPage")
);
const MaskingRulesPage = React.lazy(
  () => import("@/pages/rules/MaskingRulesPage")
);
const PendingRulesPage = React.lazy(
  () => import("@/pages/learning/PendingRulesPage")
);
const TrainingDataPage = React.lazy(
  () => import("@/pages/learning/TrainingDataPage")
);
const ModelsPage = React.lazy(() => import("@/pages/models/ModelsPage"));
const APIKeysPage = React.lazy(
  () => import("@/pages/api-management/APIKeysPage")
);
const AccessRequestsPage = React.lazy(
  () => import("@/pages/api-management/AccessRequestsPage")
);
const APIUsagePage = React.lazy(
  () => import("@/pages/api-management/APIUsagePage")
);
const UsersPage = React.lazy(() => import("@/pages/users/UsersPage"));
const AuditPage = React.lazy(() => import("@/pages/audit/AuditPage"));
const SettingsPage = React.lazy(
  () => import("@/pages/settings/SettingsPage")
);
const ProfilePage = React.lazy(
  () => import("@/pages/profile/ProfilePage")
);

const PageLoader: React.FC = () => (
  <div
    style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "100vh",
    }}
  >
    <Spin size="large" />
  </div>
);

const App: React.FC = () => {
  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* Protected routes */}
        <Route
          element={
            <ProtectedRoute>
              <AppShell />
            </ProtectedRoute>
          }
        >
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/detection" element={<DetectionPage />} />
          <Route path="/taxonomy" element={<TaxonomyPage />} />

          {/* Rules */}
          <Route path="/rules/regex" element={<RegexRulesPage />} />
          <Route path="/rules/field-patterns" element={<FieldPatternsPage />} />
          <Route path="/rules/context" element={<ContextRulesPage />} />
          <Route path="/rules/masking" element={<MaskingRulesPage />} />

          {/* Learning */}
          <Route
            path="/learning/pending-rules"
            element={<PendingRulesPage />}
          />
          <Route
            path="/learning/training-data"
            element={<TrainingDataPage />}
          />

          {/* Models */}
          <Route path="/models" element={<ModelsPage />} />

          {/* API Management */}
          <Route path="/api/keys" element={<APIKeysPage />} />
          <Route path="/api/requests" element={<AccessRequestsPage />} />
          <Route path="/api/usage" element={<APIUsagePage />} />

          {/* Admin */}
          <Route path="/users" element={<UsersPage />} />
          <Route path="/audit" element={<AuditPage />} />

          {/* Settings & Profile */}
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/settings/:tab" element={<SettingsPage />} />
          <Route path="/profile" element={<ProfilePage />} />
        </Route>

        {/* Default redirect */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Suspense>
  );
};

export default App;
